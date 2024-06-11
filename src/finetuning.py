import random
import numpy as np
from collections import defaultdict
import pandas as pd
import torch
import transformers
from pathlib import Path
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from huggingface_hub import login
from peft import LoraConfig  # LORA: low-rank adaptation
from peft import get_peft_model  # PEFT: parameter-efficient fine-tuning
from peft import prepare_model_for_kbit_training
from accelerate import FullyShardedDataParallelPlugin, Accelerator, PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from chat_template import ChatTemplate
from transformers import EarlyStoppingCallback

def formatting_func(record):
    return record['text']

if __name__ == '__main__':
    import argparse

    # LOOK AT THE SBATCH FILES TO SEE OUR EXPERIMENTED PARAMETERS
    parser = argparse.ArgumentParser(
        prog='Finetuning',
        description="Finetuning a Causal LLM to generate coherent definitions",
        epilog='Finetuning a Causal LLM')
    parser.add_argument('-M', '--max_seq_length', default=100, type=int)
    parser.add_argument('--qlora', default="False", type=str)
    parser.add_argument('--lora_rank', default=128, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=2.5e-5, type=float)
    parser.add_argument('--eval_steps', default=500, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
    parser.add_argument('--num_train_epochs', default=5, type=int)
    parser.add_argument('--warmup_ratio', default=0.03, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-m', '--model', default='meta-llama/Llama-2-7b-chat-hf', type=str)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-t', '--train_folder', default='datasets', type=str)
    parser.add_argument('-d', '--dev_folder', default='datasets', type=str)
    parser.add_argument('-H', '--hf_token', default="YOUR-HF-TOKEN", type=str)
    parser.add_argument('-c', '--hf_cache_dir', help='Hugginface cache directory',
                        default='my_cache', type=str)
    args = parser.parse_args()

    print(args.qlora)
    
    # login
    login(args.hf_token)

    # create chat template helper
    ct = ChatTemplate(args.model, args.hf_token, args.hf_cache_dir)

    # We do not use Urban and Wikipedia for training
    filter_dataset = lambda x: 'slang' not in x and 'wiki' not in x
    
    # load dataset
    data_sets = dict()
    data_sets['dev'] = load_dataset('json', data_files=[str(filename) for filename in Path(args.train_folder).glob('*_valid.jsonl') if filter_dataset(str(filename))], split='train').shuffle(seed=42)
    data_sets['dev'] = ct.apply_chat_template(data_sets['dev'], add_generation_prompt=False)
    data_sets['train'] = load_dataset('json', data_files=[str(filename) for filename in Path(args.train_folder).glob('*_train.jsonl') if filter_dataset(str(filename))], split='train').shuffle(seed=42)
    data_sets['train'] = ct.apply_chat_template(data_sets['train'], add_generation_prompt=False)
    
    # See: https://huggingface.co/docs/accelerate/v0.11.0/en/fsdp
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False), )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    load_parameters = dict(device_map='auto', cache_dir=args.hf_cache_dir)


    # Quantization
    if eval(args.qlora):
        print('QLORA')
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # enables double quantization (speed-up finetuning)
            bnb_4bit_quant_type="nf4",  # specifies the type of 4-bit quantization.
            bnb_4bit_compute_dtype=torch.bfloat16,  # specifies the data type for computation
        )
        load_parameters['quantization_config'] = bnb_config

        
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_parameters)
                                                 
    model.config.use_cache = False  # when fine-tuning the model, we want to update parameters (not to use the cached ones)

    # See: https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing
    model.gradient_checkpointing_enable()  # This will reduce GPU memory but slow down the process
    model = prepare_model_for_kbit_training(model)

    # Lora config
    config = LoraConfig(
        r=args.lora_rank,  # Lora attention dimension (''rank'')
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                        "lm_head",
                        ],
        bias="none",
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Parameter-Efficient Fine-Tuning
    model = get_peft_model(model, config)  # initialize peft finetuning with lora parameter

    # Apply the accelerator
    model = accelerator.prepare_model(model, device_placement=True)

    # set log dir
    output_dir = Path(args.output_dir)
    logs_dir = str(output_dir.parent) + f'/log_{output_dir.name}'

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="right",
        add_eos_token=False,  # end of sequence
        add_bos_token=False,  # beginning of sequence
        cache_dir=args.hf_cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token

    # set trainer
    trainer = SFTTrainer(
        model=model,
        dataset_text_field="text",
        eval_packing=True,
        packing=True,
        formatting_func=formatting_func,
        train_dataset=data_sets['train'],
        eval_dataset=data_sets['dev'],
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            logging_dir=logs_dir,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            do_eval=True,
            bf16=True,
            overwrite_output_dir=True,
            #metric_for_best_model="eval_loss",
            #load_best_model_at_end=True,
            evaluation_strategy="epoch",
            #evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            optim="paged_adamw_8bit",
            save_strategy="steps",
            #save_strategy="steps",  # Save the model checkpoint every logging step
            #save_steps=args.eval_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # See: https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260/5
            gradient_checkpointing=True,
            max_steps=-1,  # Stop training after this number of steps. Disabled by default (-1)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        #callbacks = [EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)]
    )

    # training!
    trainer.train()
    trainer.model.save_pretrained(args.output_dir + f'/final-epoch{args.num_train_epochs}')
    trainer.tokenizer.save_pretrained(args.output_dir + f'/final-epoch{args.num_train_epochs}')
    pd.DataFrame(trainer.state.log_history).to_csv(args.output_dir + f'/log.tsv', sep='\t', index=False)

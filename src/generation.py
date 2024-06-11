import torch
from tqdm import tqdm
from pathlib import Path
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import login
from chat_template import ChatTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def formatting_func(record):
    return record['text']

def tokenization(dataset):
    result = tokenizer(formatting_func(dataset),
                       truncation=True,
                       max_length=args.max_length,
                       padding="max_length",
                       add_special_tokens=False)
    return result


if __name__ == '__main__':
    import argparse

    # LOOK AT THE SBATCH FILES TO SEE OUR EXPERIMENTED PARAMETERS
    parser = argparse.ArgumentParser(
        prog='Gloss generation',
        description="Generating gloss by using the finetuned LLM",
        epilog='Gloss generation through LLMs')
    parser.add_argument('-m', '--model', default='meta-llama/Meta-Llama-3-8B-Instruct', type=str)
    parser.add_argument('--qlora', default="False", type=str)
    parser.add_argument('-M', '--max_length', default=100, type=int)
    parser.add_argument('-c', '--hf_cache_dir', help='Hugginface cache directory',
                        default='my_cache', type=str)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-f', '--finetuned_model',
                        type=str)
    parser.add_argument('-t', '--test_set', type=str)
    parser.add_argument('-H', '--hf_token', default="[YOUR-TOKEN-HERE]", type=str)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    # login
    login(args.hf_token)

    # create chat template helper
    ct = ChatTemplate(args.model, args.hf_token, args.hf_cache_dir)


    load_parameters = dict(device_map='auto', cache_dir=args.hf_cache_dir)


    # Quantization
    if eval(args.qlora):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # enables double quantization (speed-up finetuning)
            bnb_4bit_quant_type="nf4",  # specifies the type of 4-bit quantization.
            bnb_4bit_compute_dtype=torch.bfloat16,  # specifies the data type for computation
        )
        load_parameters['quantization_config'] = bnb_config

    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_parameters)
    

    # load finetuned model
    ft_model = PeftModel.from_pretrained(model, args.finetuned_model)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        add_eos_token=True,  # end of sequence
        add_bos_token=True,  # beginning of sequence
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # load dataset
    data_sets = dict()
    test_dataset = load_dataset('json', data_files=args.test_set, split='train')
    test_dataset = ct.apply_chat_template(test_dataset, add_generation_prompt=True)

    # tokenize
    tokenized_test_dataset = test_dataset.map(tokenization)

    ft_model.eval()

    eos_tokens = list()
    eos_tokens.append(tokenizer.encode(';', add_special_tokens=False)[0])
    eos_tokens.append(tokenizer.encode(' ;', add_special_tokens=False)[0])
    eos_tokens.append(tokenizer.encode('.', add_special_tokens=False)[0])
    eos_tokens.append(tokenizer.encode(' .', add_special_tokens=False)[0])
    eos_tokens.append(tokenizer.eos_token_id)

    output = list()
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_test_dataset), args.batch_size)):
            batch = tokenized_test_dataset[i:i + args.batch_size]

            model_input = dict()

            for k in ['input_ids', 'attention_mask']:
                model_input[k] = torch.tensor(batch[k]).to('cuda')

            output_ids = ft_model.generate(**model_input,
                                           max_length=args.max_length * args.batch_size,
                                           forced_eos_token_id=eos_tokens,
                                           max_time=4.5 * args.batch_size,
                                           eos_token_id=eos_tokens,
                                           temperature=0.00001,
                                           pad_token_id=tokenizer.eos_token_id)

            answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for j, answer in enumerate(answers):
                try:
                    if 'llama3' in args.finetuned_model:
                        answer = answer.split('\n')[-1].strip(" .,;:")
                    elif 'llama2' in args.finetuned_model:
                        answer = answer.split('[/INST]')[-1].strip(" .,;:")
                        if 'SYS>>' in answer:
                            answer=''
                except:
                    print('-- Exception: --')
                    print(f'batch: {i}th')
                    print(f'answer: {j}th')
                    print(answer)
                    answer = ''
                output.append(answer.replace('\n', ' ') + '\n')

    # store answers
    filename = Path(args.output)
    folder = filename.parent
    folder.mkdir(parents=True, exist_ok=True)

    with open(filename, 'w') as f:
        f.writelines(output)

import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

parser = argparse.ArgumentParser(prog='Gloss generation',
                                 description="Generating gloss by using the Flan-T5",
                                 epilog='Gloss generation through LLMs')
parser.add_argument('-b', '--batch_size', default=4, type=int)
parser.add_argument('-m', '--model_name', default="flan-t5-definition-en-xl", type=str)
parser.add_argument('-o', '--output_folder', default="t5-answers", type=str)
parser.add_argument('-t', '--test_set', type=str)
args = parser.parse_args()


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(f'ltg/{args.model_name}', add_prefix_space=True)

# load model
model = AutoModelForSeq2SeqLM.from_pretrained(f'ltg/{args.model_name}', device_map="auto")


# load test set
test_dataset = load_dataset('json', data_files=args.test_set, split='train')


# Prompt used in Giulianelli et al., 2023
input_sentences = list()
for row in test_dataset:
    prompt = f"{row['example']} What is the definition of {row['target']}?"
    input_sentences.append(prompt)


tokenized_input = tokenizer(input_sentences,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=256)

target_ids = tokenizer(test_dataset['target'], add_special_tokens=False).input_ids
target_ids = torch.tensor([el[-1] for el in target_ids])

tokenized_input = tokenized_input.to("cuda")
target_ids = target_ids.to("cuda")

test_tensor_dataset = torch.utils.data.TensorDataset(tokenized_input["input_ids"], 
                                              tokenized_input["attention_mask"],
                                              target_ids)

test_iter = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=4, shuffle=False)
gen_args = dict(do_sample=0, num_beams=1, num_beam_groups=1, temperature=0.00001, repetition_penalty=1.0)

# generate definitions
definitions = []
for inp, att, targetwords in tqdm(test_iter):
    bad = [[el] for el in targetwords.tolist()]
    outputs = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=60, bad_words_ids=bad, **gen_args)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    definitions += predictions

# store definitions
Path(f'{args.output_folder}/{args.model_name}/').mkdir(parents=True, exist_ok=True)
with open(f'{args.output_folder}/{args.model_name}/{args.test_set.split("/")[-1].replace(".jsonl", ".txt")}', mode='w', encoding='utf-8') as f:
    f.writelines([d + '\n' for d in definitions])

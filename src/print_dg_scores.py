import pandas as pd

for dataset in ['wordnet', 'en-codwoe', 'oxford', 'slang', 'wiki']:
    try:
        print(f'-- Llama3Instruct: {dataset.upper()} --')
        df = pd.read_csv(f'lora-evaluation/llama3instruct-512-1024-0.05-0.001-1e-4/{dataset}_test.tsv', sep='\t')
        for metric in ['rougeL', 'nltk_bleu', 'bertscore', 'nist', 'sacrebleu', 'meteor', 'exact_match']:
            print(metric, df[metric].mean().round(3))
        print()
    except:
        pass

    try:
        print(f'-- Llama2Chat: {dataset.upper()} --')
        df = pd.read_csv(f'lora-evaluation/llama2chat-1024-2048-0.1-0.001-1e-4/{dataset}_test.tsv', sep='\t')
        for metric in ['rougeL', 'nltk_bleu', 'bertscore', 'nist', 'sacrebleu', 'meteor', 'exact_match']:
            print(metric, df[metric].mean().round(3))
        print()
    except:
        pass

    try:
        print(f'-- T5: {dataset.upper()} --')
        df = pd.read_csv(f't5-evaluation/flan-t5-definition-en-xl/{dataset}_test.tsv', sep='\t')
        for metric in ['rougeL', 'nltk_bleu', 'bertscore', 'nist', 'sacrebleu', 'meteor', 'exact_match']:
            print(metric, df[metric].mean().round(3))
        print()
    except:
        pass

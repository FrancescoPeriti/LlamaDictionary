import csv
import json
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

def process_codwoe():
    def processing(x):
        return x[0].upper() + x[1:]
    
    partitions = ['train', 'dev', 'test']

    for lang in ['en']:
        df = pd.read_csv(f"data/raw_data/CoDWoE/full_dataset/{lang}.complete.csv", sep=',')
        gloss2word = {row['gloss']: dict(target=row['word'], gloss=row['gloss'], example=row['example']) for _, row in df.iterrows()}

        for partition in ['train', 'test', 'dev']:
            if partition == 'test':
                folder='test'
                filename = f'data/raw_data/CoDWoE/{folder}-data_all/{lang}.{partition}.revdict.json'
            else:
                folder='train'
                filename = f'data/raw_data/CoDWoE/{folder}-data_all/{lang}.{partition}.json'

            records = list()

            with open(filename, mode='r', encoding='utf-8') as f:
                rows = json.load(f)
            for row in rows:
                records.append(gloss2word[row['gloss']])

            Path('datasets').mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(records).drop_duplicates()
            df['gloss'] = [processing(x) for x in df['gloss']]
            df['example'] = [processing(x) for x in df['example']]

            partition = 'valid' if partition == 'dev' else partition
            df.to_json(f'datasets/{lang}-codwoe_{partition}.jsonl', orient='records', lines=True)
        

def process(sources=('oxford', 'wordnet', 'slang', 'wiki')):

    partitions = ['train', 'test', 'valid']

    for data in sources:
        records = defaultdict(list)

        for partition in partitions:
            df_defs = pd.read_csv(f'data/raw_data/data/{data}/{partition}.txt',
                                  delimiter="\t",
                                  quoting=csv.QUOTE_NONE,
                                  encoding="utf-8", on_bad_lines="warn",
                                  names=["sense", "ignore1", "ignore2", "gloss", "ignore3", "ignore4"])

            df = pd.read_csv(f'data/raw_data/data/{data}/{partition}.eg',
                             delimiter="\t",
                             quoting=csv.QUOTE_NONE,
                             encoding="utf-8",
                             on_bad_lines="warn",
                             names=["sense", "context"])

            df["target"] = [w.split("%")[0] for w in df.sense]
            df["gloss"] = df_defs.gloss


            contexts = [ctxt.replace("<TRG>", targetword).strip()
                        for ctxt, targetword in zip(df.context, df.target)]
            df["real_context"] = contexts

            for i, row in df.iterrows():
                example = row['real_context'].strip()
                target = row['target']
                gloss = str(row['gloss']).strip()

                if gloss.isnumeric() or len(gloss) < 5 or target == '' or target.isnumeric():
                    continue

                gloss = gloss[0].upper() + gloss[1:]
                example = example[0].upper() + example[1:]

                # dict to txt line
                record = dict(target=target, gloss=str(gloss), example=example)
                records[partition].append(record)

        for partition in records:
            Path('datasets').mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records[partition]).drop_duplicates().to_json(f'datasets/{data}_{partition}.jsonl', orient='records', lines=True)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data processing')
    parser.add_argument('-d', '--dataset',
                        nargs='+', default=['codwoe', 'oxford', 'wordnet', 'slang', 'wiki'],
                        type=str,  help='List of datasets')
    args = parser.parse_args()
    
    # process datasets
    process(sources=[d for d in args.dataset if d != 'codwoe'])

    # process codwoe
    if 'codwoe' in args.dataset: process_codwoe()

import json
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str, )
args = parser.parse_args()

# load dataset
dataset = load_dataset('csv', column_names=['target', 'pos', 'indexes', 'example1', 'example2'], 
                       sep='\t', 
                       data_files=args.filename, 
                       split='train')

# we tried by quotig the target word as the pos infomation was available
def processing(example):
    text = example['example']
    start = int(example['start'])
    end = int(example['end'])
    return text[:start] + '"' + text[start:end] + '"' + text[end:]

rows = list()
for row in dataset:
    indexes=[int(i) for i in row['indexes'].split('-')]
    token1 = row['example1'].split()[indexes[0]]
    token2 = row['example2'].split()[indexes[1]]
    before1 = len(" ".join(row['example1'].split()[:indexes[0]])) + (1 if indexes[0] > 0 else 0)
    before2 = len(" ".join(row['example2'].split()[:indexes[1]])) + (1 if indexes[1] > 0 else 0)

    record = processing(dict(target=row['target'], example=row['example1'], start=before1, end=before1+len(token1)))
    rows.append(json.dumps(dict(target=row['target'], example=record, start=before1, end=before1+len(token1)))+'\n')

    record = processing(dict(target=row['target'], example=row['example2'], start=before2, end=before2+len(token2)))
    rows.append(json.dumps(dict(target=row['target'], example=record, start=before2, end=before2+len(token2)))+'\n')

with open(args.filename, mode='w', encoding='utf-8') as f:
    f.writelines(rows)

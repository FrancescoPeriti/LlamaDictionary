import os
import csv
import pandas as pd
from pathlib import Path
from datasets import load_datasets

# we tried by quotig the target word as the pos infomation was available
def processing(example, length=None):
    try:
        start, end = example['indexes_target_token'].split(':')
        start, end = int(start), int(end)
    except:
        start = example['indexes_target_token'].split(':')[0]
        end = len(example['example'])

    example['example'] = example['example'][:int(start)] + '"' + example['example'][int(start):int(end)] + '"' + example['example'][int(end):]

    if length is None:
        return example['example']
    else:
        return example['example'][max(0,int(start)-length):int(end)+length] # limit context length


targets = sorted(os.listdir('dwug_en/data/'))

dfs = dict()
for target in targets:
    df = pd.read_csv(f'dwug_en/data/{target}/uses.csv', sep='\t', quoting=csv.QUOTE_NONE)
    clusters = pd.read_csv(f'dwug_en/clusters/opt/{target}.csv', sep='\t')
    uses_to_remove = clusters[clusters['cluster']==-1].identifier.values # this (noisy) usages have been ignored for SemEval-English benchmark creation
    df = df[~df.identifier.isin(uses_to_remove)].merge(clusters)
    df = df[['grouping', 'identifier', 'context', 'cluster', 'indexes_target_token']]
    df = df.rename(columns={'context': 'example'})
    target = target.replace('_nn', '').replace('_vb', '')
    df['target'] = target
    dfs[target] = df

    # create directories
    Path('dwug_en/wsi/').mkdir(parents=True, exist_ok=True)
    Path('dwug_en25/wsi/').mkdir(parents=True, exist_ok=True)
    Path('dwug_en50/wsi/').mkdir(parents=True, exist_ok=True)
    Path('dwug_en75/wsi/').mkdir(parents=True, exist_ok=True)
    Path('dwug_en100/wsi/').mkdir(parents=True, exist_ok=True)

    # store datasets 
    dfs[target].to_json(f'dwug_en/wsi/{target}.jsonl', orient='records', lines=True)
    test_dataset = load_dataset('json', data_files=f'dwug_en/wsi/{target}.jsonl', split='train')
    df['example'] = test_dataset.map(lambda x: {'example': processing(x)})['example']
    df.to_json(f'dwug_en/wsi/{target}.jsonl', orient='records', lines=True)

    # store datasets with limited contexts (25 chars)
    test_dataset = load_dataset('json', data_files=f'dwug_en/wsi/{target}.jsonl', split='train')
    df['example'] = test_dataset.map(lambda x: {'example': processing(x, 25)})['example']
    df.to_json(f'dwug_en/wsi25/{target}.jsonl', orient='records', lines=True)

    # store datasets with limited contexts (50 chars)
    test_dataset = load_dataset('json', data_files=f'dwug_en/wsi/{target}.jsonl', split='train')
    df['example'] = test_dataset.map(lambda x: {'example': processing(x, 50)})['example']
    df.to_json(f'dwug_en/wsi50/{target}.jsonl', orient='records', lines=True)

    # store datasets with limited contexts (75 chars)
    test_dataset = load_dataset('json', data_files=f'dwug_en/wsi/{target}.jsonl', split='train')
    df['example'] = test_dataset.map(lambda x: {'example': processing(x, 75)})['example']
    df.to_json(f'dwug_en/wsi75/{target}.jsonl', orient='records', lines=True)

    # store datasets with limited contexts (100 chars)
    test_dataset = load_dataset('json', data_files=f'dwug_en/wsi/{target}.jsonl', split='train')
    df['example'] = test_dataset.map(lambda x: {'example': processing(x, 100)})['example']
    df.to_json(f'dwug_en/wsi100/{target}.jsonl', orient='records', lines=True)
    
dfs_labels = dict()
for target in targets:
    df = pd.read_csv(f'dwug_en/data/{target}/judgments.csv', sep='\t')[['identifier1', 'identifier2', 'judgment']]
    df['identifier1'] = df['identifier1'].astype(str)
    df['identifier2'] = df['identifier2'].astype(str)
    target = target.replace('_nn', '').replace('_vb', '')    
    df = df.sort_values(['identifier1', 'identifier2']).groupby(['identifier1', 'identifier2']).mean().reset_index()
    df = df[(df['identifier1'].isin(dfs[target].identifier.values)) & (df['identifier2'].isin(dfs[target].identifier.values))]
    Path('dwug_en/wic/').mkdir(parents=True, exist_ok=True)
    df.to_json(f'dwug_en/wic/{target}.jsonl', orient='records', lines=True)

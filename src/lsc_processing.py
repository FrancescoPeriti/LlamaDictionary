import os
import csv
import pandas as pd
from pathlib import Path

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
    dfs[target].to_json(orient='records', lines=True)
    Path('dwug_en/wsi/').mkdir(parents=True, exist_ok=True)
    dfs[target].to_json(f'dwug_en/wsi/{target}.jsonl', orient='records', lines=True)
    
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

import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--qlora', action='store_true')
parser.add_argument('-m', '--metric', default='nltk_bleu')
args = parser.parse_args()

# parameters
if args.qlora: 
    folder = 'qlora-evaluation'
else:
    folder = 'lora-evaluation' 

metric=args.metric

df = list()
for filename in Path(f'{folder}/').rglob('*/*_test.tsv'):
    model_name = filename.parent.name
    model, lora_rank, lora_alpha, dropout, wd, lr = model_name.split('-')[:-2] + ["-".join(model_name.split('-')[-2:])]
    
    # load performance
    df_eval = pd.read_csv(str(filename), sep='\t')

    # create record
    record = dict(dataset=filename.name.replace('.tsv', ''),
                  model=model,
                  lora_rank=lora_rank,
                  lora_alpha=lora_alpha,
                  dropout=dropout,
                  wd=wd,
                  lr=lr)

    # get average
    for c in df_eval.columns[4:]:
        record[c] = df_eval[c].mean().round(4)

    # append
    df.append(record)

# create dataframe
df = pd.DataFrame(df)

df.lora_rank = df.lora_rank.astype(float)
df.dropout = df.dropout.astype(float)
df.bertscore = df.bertscore.astype(float)

# Train dataset
df = df.rename(columns={'dataset':'Datasets'})
df = df[df['Datasets'].isin(['oxford_test', 'en-codwe_test', 'wordnet_test'])]
dataset_names = list()
for i in df['Datasets']:
    if i=='wordnet_test':
        dataset_names.append('WordNet')
    elif i == 'oxford_test':
        dataset_names.append('Oxford')
    else:
        dataset_names.append('CoDWoe')
df['Datasets'] = dataset_names

# lama2-3 split
df_2 = df[df['model'] == 'llama2chat']
df_3 = df[df['model'] == 'llama3instruct']
df_2 = df_2[['Datasets', 'lora_rank', 'lora_alpha', 'rougeL', 'nltk_bleu', 'nist',  'sacrebleu', 'meteor', 'bertscore', 'exact_match']]
df_3 = df_3[['Datasets', 'lora_rank', 'lora_alpha', 'rougeL', 'nltk_bleu', 'nist',  'sacrebleu', 'meteor', 'bertscore', 'exact_match']]
df_2 = df_2
df_3 = df_3

# Initialize the plot and palette
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 2.5))
dataset_colors = {
    'Oxford': '#ff7f0e',
    'WordNet': '#2ca02c',
    'CoDWoe': '#1f77b4' 
}

for i, df_ in enumerate([df_2, df_3]):
    # assign id to rank values
    rank2id = {int(v):k for k,v in dict(enumerate(sorted(df_['lora_rank'].unique()))).items()}

    # x values
    df_['x'] = [rank2id[int(i)] for i in df_['lora_rank']]

    xticks = [0] + sorted(df_['x'].unique())
    sns.lineplot(data=df_, x='x', y=metric, hue='Datasets', marker='o', ax=axes[i], palette=dataset_colors)
    axes[i].set_xticks(xticks)
    axes[i].set_xticklabels([''] + [f'r: {str(int(i))}\nα: {str(int(i*2))}' for i in sorted(df_['lora_rank'].unique())])
    if metric=='bertscore':
        axes[i].set_ylabel('BERT-F1')
    elif metric=='nltk_bleu':
        axes[i].set_ylabel('BLEU')
    axes[i].set_xlabel('')

    title = 'LoRA - ' if not folder.startswith('q') else 'QLoRA - '
    if i == 0:
        axes[i].set_title(title+'Llama2Dictionary')
    else:
        axes[i].set_title(title+'Llama3Dictionary')
    
plt.tight_layout()
plt.savefig(f'{folder}-{args.metric}.png')
plt.show()

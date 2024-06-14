import pandas as pd
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt

# load data
df = load_dataset('json', data_files=f'wsi-lsc.jsonl', split='train').to_pandas()

# filter considered generative models
df = df[df['ft_model_name'].isin(['llama2chat-1024-2048-0.1-0.001-1e-4', 
                          'llama3instruct-512-1024-0.05-0.001-1e-4', 
                          'flan-t5-definition-en-xl'])]
# filter considered SBERT model 
df = df[(df['model'] == 'all-distilroberta-v1')]
df['suffix'] = [int(i) if i!='' else 150 for i in df['suffix']]
df['filter_'] = [int(i) for i in df['filter_']]

model_names = list()
for m in df.ft_model_name.values:
    if m == 'flan-t5-definition-en-xl':
        model_names.append('Flan-T5')
    elif m == 'llama2chat-1024-2048-0.1-0.001-1e-4':
        model_names.append('Llama2Dict.')
    else:
        model_names.append('Llama3Dict.')
df['ft_model_name'] = model_names

df = df[['suffix', 'filter_', 'ft_model_name', 'APD', 'APDP']]

# filter_ is fixed to 1.
df_tmp = df[df['filter_'] == 1]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 2.5))
axes[0] = sns.lineplot(data=df_tmp, x='suffix', y='APD', marker='o', label='APD', markersize=5, ax=axes[0])
axes[0] = sns.lineplot(data=df_tmp, x='suffix', y='APDP', marker='o', label='APDP', markersize=5, ax=axes[0])
axes[0].set_xticks([25, 50, 75, 100, 150])
axes[0].set_xticklabels(['50', '100', '150', '200', 'NoLimit'])
axes[0].set_xlabel('Sentence length')
axes[0].set_ylabel('Spearman Corr.')

# suffix is fixed to 100. (100 left and 100 right=200).
df_tmp = df[df['suffix'] == 100]
axes[1] = sns.lineplot(data=df_tmp, x='filter_', y='APD', marker='o', label='APD', markersize=5, ax=axes[1])
axes[1] = sns.lineplot(data=df_tmp, x='filter_', y='APDP', marker='o', label='APDP', markersize=5, ax=axes[1])
axes[1].set_xticks([1, 2, 3, 4])
axes[1].set_xticklabels(['1', '2', '3', '4'])
axes[1].set_xlabel('Word removal')
axes[1].set_ylabel('Spearman Corr.')

#plot.legend(title='Metrics')
plt.title('Lexical Semantic Change')
plt.tight_layout()
plt.savefig('lsc.png')

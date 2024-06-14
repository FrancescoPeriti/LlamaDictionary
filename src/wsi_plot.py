import pandas as pd
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt

# load data
df = load_dataset('json', data_files=f'wsi-lsc.jsonl', split='train').to_pandas()

# filter considered models
df = df[df['ft_model_name'].isin(['llama2chat-1024-2048-0.1-0.001-1e-4', 
                                  'llama3instruct-512-1024-0.05-0.001-1e-4', 
                                  'flan-t5-definition-en-xl'])]

# filter considered sbert model
df = df[(df['model'] == 'all-distilroberta-v1')]
df['suffix'] = [int(i) if i !='' else 150 for i in df['suffix']] # '' -> no limit (set to 150 for easier use in setting ticks) 
df['filter_'] = [int(i) for i in df['filter_']]

# melt dataframe
df = pd.melt(df, id_vars=['suffix', 'filter_', 'ft_model_name'], 
                    value_vars=['ARI', 'PUR', 'RI'], 
                    var_name='Metric', 
                    value_name='Value')

plt.figure(figsize=(3, 3))
ax = sns.boxplot(x='Metric', y='Value', data=df, palette = {'ARI': '#1f77b4', 'PUR': '#1f77b4', 'RI': '#1f77b4'})
ax.set_xlabel('WSI')
ax.set_ylabel('Clustering')
plt.xticks(rotation=0)
plt.tight_layout()
plt.title('Word Sense Induction')
plt.savefig('wsi.png')

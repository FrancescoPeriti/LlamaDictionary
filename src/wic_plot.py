import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset

# load data
df = load_dataset('json', data_files=f'wic-results.jsonl', split='train').to_pandas()

# filter considered generative models
df = df[df['model'].isin(['llama2chat-1024-2048-0.1-0.001-1e-4', 
                          'llama3instruct-512-1024-0.05-0.001-1e-4', 
                          'flan-t5-definition-en-xl'])]
# replace names
df['model'] = 'SBERT(def.)'

# set palette
palette = sns.color_palette("Set2", len(df['model'].unique()))

plt.figure(figsize=(2, 3))
ax = sns.boxplot(y='accuracy',x='model',  data=df, palette=palette)
ax.set_xlabel('')
ax.set_ylabel('Accuracy')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('wic.png')

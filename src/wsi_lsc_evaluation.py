import os
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import string
import pandas as pd
from sklearn import metrics
from scipy.stats import spearmanr
from datasets import load_dataset
from collections import defaultdict
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, rand_score

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def APDP(embeddings, L):
    L1, L2 = L[:embeddings[0].shape[0]], L[embeddings[0].shape[0]:]

    # cluster centroids
    mu_E1 = np.array([embeddings[0][L1 == label].mean(axis=0) for label in np.unique(L1)])
    mu_E2 = np.array([embeddings[1][L2 == label].mean(axis=0) for label in np.unique(L2)])
    return np.mean(cdist(mu_E1, mu_E2, metric='canberra'))


# load targets
targets = sorted([target.replace('.jsonl', '') for target in os.listdir(f'dwug_en/wsi/')])
#targets = ['attack', 'circle', 'fiction', 'lane', 'pin', 'rag', 'stab', 'twist', 'bag', 'contemplation', 'gas', 'lass', 'plane', 'record', 'stroke', 'word', 'ball', 'donkey', 'graft', 'multitude', 'player', 'relationship', 'thump', 'bit', 'edge', 'head', 'ounce', 'prop', 'risk', 'tip', 'chairman', 'face', 'land', 'part', 'quilt', 'savage', 'tree']


import argparse
parser = argparse.ArgumentParser(prog='WSI-LSC', description="WSI-LSC evaluation")
parser.add_argument('-m', '--model', type=str, default="sentence-transformers/all-distilroberta-v1")
args = parser.parse_args()

# load model
model = SentenceTransformer(args.model)

ap = HDBSCAN(metric='precomputed', allow_single_cluster=True, min_cluster_size=2, cluster_selection_method='leaf')

records = list()
for suffix in ['100', '25', '50', '75', '']:
    for model_folder in ['t5-answers', 'lora-answers']:
        folder_answers = f'lsc{suffix}-{model_folder}'
        for ft_model_name in os.listdir(folder_answers):
            print(f'--Suffix \'{suffix}\' -- Folder \'{ft_model_name}\'')
            # wic and lsc scores
            scores = defaultdict(lambda: defaultdict())
            
            for target in targets:
                # load dataset
                df = load_dataset('json', data_files=f'dwug_en/wsi{suffix}/{target}.jsonl', split='train').to_pandas()
                df['gloss'] = [" ".join([i for i in line.strip().split() if len(i) > 1])
                            for line in open(f'{folder_answers}/{ft_model_name}/{target}.txt', mode='r', encoding='utf-8')]

                # filter sentence for which models didn't answer
                df = df[(df['gloss'] != '') & (df['gloss'].apply(lambda x: not x.startswith('Please provide')))]
                
                # to avoid computing embeddings for the same gloss twice, make a copy, and drop duplicates
                df_copy = df.copy()
                df = df.drop_duplicates(subset=['gloss'])
                
                # split by time
                df1 = df[(df['grouping'] == 1)]
                df2 = df[(df['grouping'] == 2)]
                
                # encode glosses into embeddings
                e1 = model.encode(df1.gloss.values)
                e2 = model.encode(df2.gloss.values)
                
                # compute APD                
                scores['APD'][target] = cdist(e1, e2, metric='cosine').mean().item() #model.similarity(e1, e2).mean().item()

                # clustering and then APDP
                e = np.concatenate([e1, e2], axis=0)
                sim = cosine_similarity(e)
                #sim = (sim - sim.mean()) / sim.std()
                ap.fit(sim)
                labels = ap.labels_

                #scores['APDP'][target] = APDP([e1, e2], labels)

                # assign labels to duplcate glosses
                glosses = df1.gloss.values.tolist() + df2.gloss.values.tolist()
                gloss2label = {gloss: labels[i] for i, gloss in enumerate(glosses)}
                df_copy['label'] = [gloss2label[gloss] for gloss in df_copy.gloss.values]
                #scores['APD'][target] = cdist(model.encode(df_copy[df_copy['grouping']==1].gloss.values),
                 #                             model.encode(df_copy[df_copy['grouping']==2].gloss.values),
                  #                            metric='cosine').mean().item()
                scores['APDP'][target] = APDP([model.encode(df_copy[df_copy['grouping']==1].gloss.values),
                                              model.encode(df_copy[df_copy['grouping']==2].gloss.values)],
                                             df_copy[df_copy['grouping']==1].label.values.tolist()+df_copy[df_copy['grouping']==2].label.values.tolist())#APDP([e1, e2], labels)
                
                scores['RI'][target] = rand_score(df_copy.cluster.values.tolist(), df_copy.label.values.tolist())
                scores['ARI'][target] = adjusted_rand_score(df_copy.cluster.values.tolist(), df_copy.label.values.tolist())
                scores['PUR'][target] = purity_score(df_copy.cluster.values.tolist(), df_copy.label.values.tolist())
                scores['n_cluster'][target] = len(set(labels))
                scores['n_true_cluster'][target] = len(set(df_copy.cluster.values))

            df_gold = pd.read_csv(f'dwug_en/stats/opt/stats_groupings.csv', sep='\t')
            df_gold['lemma'] = df_gold['lemma'].apply(lambda x: x.replace('_vb', '').replace('_nn', ''))
            df_gold = df_gold[['lemma', 'change_graded']]
            df_gold = df_gold[df_gold['lemma'].isin(targets)]
            
            record = dict(suffix=suffix, model_folder=model_folder, ft_model_name=ft_model_name, model=args.model)
            corr, _ = spearmanr(df_gold.change_graded.values, [scores['APD'][target] for target in df_gold.lemma])
            record['APD']=round(corr, 3)
            corr, _ = spearmanr(df_gold.change_graded.values, [scores['APDP'][target] for target in df_gold.lemma])
            record['APDP'] = round(corr, 3)
            for metric in ['ARI', 'RI', 'PUR']:
                record[metric] = np.array([scores[metric][target] for target in df_gold.lemma]).mean().round(3)
            for column in ['n_cluster', 'n_true_cluster']:
                record[column] = [scores[column][target] for target in df_gold.lemma]

            records.append(record)
            print(record)

with open('wsi-lsc.jsonl', mode='a+', encoding='utf-8') as f:
    for row in records:
        f.write(json.dumps(row)+'\n')

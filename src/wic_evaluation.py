import os
import json
import string
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, roc_curve

def set_threshold(y_true: np.array, y: np.array) -> float:
    """
    Find the threshold that maximize accuracy on the Dev set.

    Args:
        y(np.array): array containing predicted values
        y_true(np.array): array containing ground truth values.
    Returns:
        thr
    """

    # False Positive Rate - True Positive Rate - Thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y)

    scores = []
    for thr in thresholds:
        scores.append(accuracy_score(y_true, [pred >= thr for pred in y])) 
    scores = np.array(scores)

    # Max accuracy
    max_ = scores.max()

    # Threshold associated to the maximum accuracy
    max_threshold = thresholds[scores.argmax()]

    return max_threshold
    

# load model
model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

# load examples
dev_examples = [json.loads(i) for i in open('wic/dev/dev.data.txt').readlines()]
test_examples = [json.loads(i)  for i in open('wic/test/test.data.txt').readlines()]

results = list()
for folder_answers in ['wic-lora-anwers', 'wic-t5-answers']:
    print('--', folder_answers.upper(), '--')
    
    for ft_model_name in os.listdir(folder_answers):
        # load model answers
        dev_answers = [i.strip() for i in open(f'{folder_answers}/{ft_model_name}/dev.data.txt').readlines()]
        test_answers = [i.strip() for i in open(f'{folder_answers}/{ft_model_name}/test.data.txt').readlines()]

        # load labels
        dev_labels = [int(i.strip() == 'T') for i in open('wic/dev/dev.gold.txt').readlines()]
        test_labels = [int(i.strip() == 'T') for i in open('wic/test/test.gold.txt').readlines()]

        # encode example into embeddings
        emb_dev = model.encode(dev_answers)
        sim_dev = model.similarity(emb_dev, emb_dev) 
        emb_test = model.encode(test_answers)
        sim_test = model.similarity(emb_test, emb_test)

        # Dev and Test sets
        dev = [dict(gloss1=dev_answers[i], gloss2=dev_answers[i + 1],
                    sim=sim_dev[i][i+1], example1=dev_examples[i]['example'], example2=dev_examples[i+1]['example'],
                    label=dev_labels[i//2]) for i in range(0, len(dev_answers) - 1, 2)]
    
        test = [dict(gloss1=test_answers[i], gloss2=test_answers[i + 1], sim=sim_test[i][i+1],
                     example1=test_examples[i]['example'], example2=test_examples[i+1]['example'],
                     label=test_labels[i//2]) for i in range(0, len(test_answers) - 1, 2)]


        # get best threshold on Dev
        thr = set_threshold(np.array(dev_labels), np.array([row['sim'] for row in dev]))  # dev

        # use thresold for predictions on Test
        test_preds = [int(i >= thr) for i in np.array([row['sim'] for row in test])]  # test

        # compute accuracy
        acc_test = accuracy_score(np.array(test_labels), test_preds)
        results.append(dict(model=ft_model_name, accuracy=round(acc_test, 3)))
        print('MODEL:', ft_model_name, ' - ACCURACY:', round(acc_test, 3))

with open('wic-results.jsonl', mode='w', encoding='utf-8') as f:
    for row in results:
        f.write(json.dumps(row)+'\n')

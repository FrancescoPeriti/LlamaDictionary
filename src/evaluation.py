import json
import os
import string
import random
import evaluate
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
from nltk.translate import bleu_score, nist_score

random.seed(42)

# remove punctuation
def preprocessing(text):
    text = text.strip()
    # return text
    #text = text[0].upper() + text[1:]
    translator = str.maketrans('', '', string.punctuation)
    return " ".join(text.lower().translate(translator).split()).strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_folder", type=str, )
    parser.add_argument("--metrics", nargs='*', )
    parser.add_argument("-t", "--test_set")
    parser.add_argument("-a", "--answers")
    args = parser.parse_args()

    output_name = os.path.basename(args.test_set).replace('.jsonl', '.tsv').replace('.txt', '.tsv')
    
    # load answers
    answers = [line.strip() for line in open(args.answers, mode='r', encoding='utf-8')]
    
    # load dataset
    test_dataset = load_dataset('json', data_files=args.test_set, split='train')
    test_dataset = test_dataset.add_column("answer", answers)
    test_dataset = test_dataset.to_pandas()

    test_dataset['gloss'] = test_dataset['gloss'].apply(preprocessing)
    test_dataset['answer'] = test_dataset['answer'].apply(preprocessing)

    # metrics
    if not args.metrics:
        args.metrics = ["rougeL", "nltk_bleu", "nist", "sacrebleu", "meteor", "bertscore", "exact_match"]

    eval = {
        "rougeL": (evaluate.load("rouge"), "rougeL"),
        "meteor": (evaluate.load("meteor"), "meteor"),
        "bertscore": (evaluate.load("bertscore"), "f1"),
        "sacrebleu": (evaluate.load("sacrebleu"), "score"),
        "exact_match": (evaluate.load("exact_match"), "exact_match"),
    }


    # row evaluation
    results = defaultdict(list)
    for _, row in tqdm(test_dataset.iterrows(), total=test_dataset.shape[0]):
        for metric in args.metrics:
            gold_gloss, pred_gloss = row['gloss'], row['answer']

            if metric == "nltk_bleu":
                auto_reweigh = False if len(pred_gloss.split()) == 0 else True
                results[metric].append(bleu_score.sentence_bleu([gold_gloss.split()], pred_gloss.split(),
                                                                     smoothing_function=bleu_score.SmoothingFunction().method2,
                                                                     auto_reweigh=auto_reweigh))
                                
            elif metric == "nist":
                n = 5
                pred_len = len(pred_gloss.split())
                if pred_len < 5:
                    n = pred_len
                try:
                    results[metric].append(nist_score.sentence_nist([gold_gloss.split()], pred_gloss.split(), n=n))
                except:
                    results[metric].append(0)

            elif metric == "bertscore":
                evaluator, output_key = eval[metric]
                results[metric].append(evaluator.compute(predictions=[pred_gloss],
                                                         references=[gold_gloss], lang="en")[output_key][0]) 
            else:
                evaluator, output_key = eval[metric]
                results[metric].append(evaluator.compute(predictions=[pred_gloss],
                                                         references=[gold_gloss])[output_key])

    # store results
    for k in results:
        test_dataset[k] = results[k]

    filename = Path(args.output_folder + '/' + output_name)
    folder = filename.parent
    folder.mkdir(parents=True, exist_ok=True)
    test_dataset.to_csv(str(filename), sep='\t', index=False)

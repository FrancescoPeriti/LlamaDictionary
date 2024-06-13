#!/bin/bash
#SBATCH -A [YOUR-PROJECT-ID] -p alvis
#SBATCH -t 5:00:00
#SBATCH --gpus-per-node=A100:1

export HF_DATASETS_CACHE=$TMPDIR
export HF_HOME=$TMPDIR
export CUDA_VISIBLE_DEVICES="0"

# params used in paper: length:1, model:all-distilroberta-v1

lengths=(1 2 3 4) 
models=("all-mpnet-base-v2" "multi-qa-mpnet-base-dot-v1" "all-distilroberta-v1" "all-MiniLM-L12-v2" "multi-qa-distilbert-cos-v1" "all-MiniLM-L6-v2" "multi-qa-MiniLM-L6-cos-v1" "paraphrase-multilingual-mpnet-base-v2" "paraphrase-albert-small-v2" "paraphrase-multilingual-MiniLM-L12-v2" "paraphrase-MiniLM-L3-v2" "distiluse-base-multilingual-cased-v1" "distiluse-base-multilingual-cased-v2")

for l in "${lengths[@]}"; do
    for model in "${models[@]}"; do
        python src/wsi_lsc_evaluation.py -m "$model" -l $l
    done
done

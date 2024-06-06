#!/bin/bash
#SBATCH -A NAISS2024-5-148 -p alvis
#SBATCH -t 1:00:00
#SBATCH --gpus-per-node=A40:1

export HF_DATASETS_CACHE=$TMPDIR
export HF_HOME=$TMPDIR

export CUDA_VISIBLE_DEVICES="0"

python src/wic_evaluation.py

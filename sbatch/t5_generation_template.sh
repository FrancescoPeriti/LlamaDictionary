#!/bin/bash
#SBATCH -A NAISS2024-5-148 -p alvis
#SBATCH -t 8:00:00
#SBATCH --gpus-per-node=A40:1

export HF_DATASETS_CACHE=$TMPDIR
export HF_HOME=$TMPDIR
export CUDA_VISIBLE_DEVICES="0"

echo "Output: $1"
echo "Model: $2"
echo "Finetuned model: $3"
echo "Test set: $4"
echo "Qlora: $5"

python src/t5_generation.py --test_set $1

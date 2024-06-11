#!/bin/bash
#SBATCH -A [YOUR-PROJECT-ID] -p alvis
#SBATCH -t 9:00:00
#SBATCH --gpus-per-node=A100fat:1

export HF_DATASETS_CACHE=$TMPDIR
export HF_HOME=$TMPDIR

export CUDA_VISIBLE_DEVICES="0"

echo "Output: $1"
echo "Model: $2"
echo "Finetuned model: $3"
echo "Test set: $4"
echo "Qlora: $5"

python src/generation.py --output $1 --model $2 --finetuned_model $3 --test_set $4 --batch_size 8 --hf_token [YOUR_HF_TOKEN] --hf_cache_dir $HF_HOME --qlora $5

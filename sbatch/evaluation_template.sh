#!/bin/bash
#SBATCH -A [YOUR-PROJECT-ID] -p alvis
#SBATCH -t 4-30:00:00
#SBATCH -C NOGPU

echo "Output: $1"
echo "Answer: $2"
echo "Dataset: $3"

export HF_DATASETS_CACHE=$TMPDIR
export HF_HOME=$TMPDIR

python src/evaluation.py --output_folder $1 --test_set $3 --answers $2

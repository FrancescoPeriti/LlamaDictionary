#!/bin/bash
#SBATCH -A [YOUR-PROJECT-ID] -p alvis
#SBATCH -t 9:00:00
#SBATCH --gpus-per-node=A100fat:1

export HF_DATASETS_CACHE=$TMPDIR
export HF_HOME=$TMPDIR

export CUDA_VISIBLE_DEVICES="0"

echo "Lora rank: $1"
echo "Lora alpha: $2"
echo "Model: $3"
echo "Output: $4"
echo "Dropout: $5"
echo "Weight decay: $6"
echo "Learning rate: $7"
echo "Qlora: $8"

python src/finetuning.py --max_seq_length 512 --lora_rank $1 --lora_alpha $2 --lora_dropout $5 --learning_rate $7 --gradient_accumulation_steps 1 --num_train_epochs 4 --warmup_ratio 0.05 --weight_decay $6 --train_folder "datasets" --dev_folder "datasets" --hf_cache_dir $HF_HOME --hf_token [YOUR_HF_TOKEN] --batch_size 32 --model $3 -o $4 --eval_steps 250 --qlora $8

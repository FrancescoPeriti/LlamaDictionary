#!/bin/bash

lora_ranks=(8 16 32 64 128 256 512 1024)
lora_alphas=(16 32 64 128 256 512 1024 2048)
dropouts=(0.05 0.1)
weight_decays=(0.001)
learning_rates=(1e-4)
lora_modes=("False" "True")

models=("meta-llama/Llama-2-7b-chat-hf" "meta-llama/Meta-Llama-3-8B-Instruct")
model_names=("llama2chat" "llama3instruct")

for ((i=0; i<${#lora_ranks[@]}; i++)); do
    lora_rank=${lora_ranks[i]}
    lora_alpha=${lora_alphas[i]}

    for ((k=0; k<${#dropouts[@]}; k++)); do
        dropout=${dropouts[k]}

        for ((h=0; h<${#weight_decays[@]}; h++)); do
            weight_decay=${weight_decays[h]}

            for ((y=0; y<${#learning_rates[@]}; y++)); do
                learning_rate=${learning_rates[y]}

                 for ((x=0; x<${#lora_modes[@]}; x++)); do
                    lora_mode=${lora_modes[x]}
                    if [ "$lora_mode" == "True" ]; then
                        lora_folder="qlora"
                    else
                        lora_folder="lora"
                    fi

                    for ((j=0; j<${#models[@]}; j++)); do
                        model=${models[j]}
                        model_name=${model_names[j]}

                        sbatch sbatch/finetuning_template.sh $lora_rank $lora_alpha $model "$lora_folder-models/$model_name-$lora_rank-$lora_alpha-$dropout-$weight_decay-$learning_rate" $dropout $weight_decay $learning_rate $lora_mode
                        echo "$lora_folder-models/$model_name-$lora_rank-$lora_alpha-$dropout-$weight_decay-$learning_rate"
                    done
                done
            done
        done
    done
done

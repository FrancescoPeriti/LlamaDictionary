#!/bin/bash

main_folder="dwug_en/wsi/"
ft_model_folder="lora-models/"
qlora="False"

model_names=("meta-llama/Llama-2-7b-chat-hf" "meta-llama/Meta-Llama-3-8B-Instruct")
ft_model_names=("llama2chat-1024-2048-0.05-0.001-1e-4" "llama3instruct-512-1024-0.05-0.001-1e-4")

for ((i=0; i<${#model_names[@]}; i++)); do
    model_name=${model_names[i]}
    ft_model_name=${ft_model_names[i]}

    # Iterate over each file in the folder
    for filename in "$main_folder"/*; do
        base_name="$(basename "$filename" .jsonl)"
        new_name="${base_name}.txt"
        output_filename="lsc-lora-answers/$ft_model_name/$new_name"
    
        echo "$output_filename"
        sbatch sbatch/generation_template.sh $output_filename $model_name "$ft_model_folder/$ft_model_name/final-epoch4" $filename $qlora
    done
done

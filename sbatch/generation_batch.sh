#!/bin/bash

model_folders=("qlora-models" "lora-models")
datasets=("en-codwoe" "oxford" "wordnet" "slang" "wiki" "stop")

# Loop through model folders
for folder in "${model_folders[@]}"; do

    # Loop through models in each folder
    for model in "$folder"/*; do

        # Loop through datasets
        for dataset in "${datasets[@]}"/*; do

            if [[ "$dataset" == 'stop/*' ]]; then
                continue
            fi

            # Define output folder, answer filename
            output_filename="${folder/models/answers}/$(basename "$model")/${dataset}_test.txt"
            ft_model_folder="${folder}/$(basename "$model")/final-epoch4/"

            # Define model
            if [[ "$model" == *"lama2chat"* ]]; then
                model_name="meta-llama/Llama-2-7b-chat-hf"
            else
                model_name="meta-llama/Meta-Llama-3-8B-Instruct"
            fi

            if [[ "$folder" == *"qlora"* ]]; then
                qlora="True"
            else
                qlora="False"
            fi

            # File does not exist and folder exists
            if [ ! -e "$output_filename" ] && [ -d "$ft_model_folder" ]; then
                sbatch sbatch/generation_template.sh $output_filename $model_name $ft_model_folder "datasets/${dataset}_test.jsonl" $qlora
                echo "$output_filename $model_name $ft_model_folder datasets/${dataset}_test.jsonl" $qlora
            fi

        done
    done
done

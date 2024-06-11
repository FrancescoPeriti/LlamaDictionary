#!/bin/bash

model_folders=("t5-answers" "qlora-answers" "lora-answers")
datasets=("oxford" "wordnet" "slang" "wiki" "en-codwoe" "stop")

# Loop through model folders
for folder in "${model_folders[@]}"; do

    # Loop through models in each folder
    for model in "$folder"/*; do

        # Loop through datasets
        for dataset in "${datasets[@]}"/*; do

            if [[ "$dataset" == 'stop/*' ]]; then
                continue
            fi

            # Define output folder, answer filename, and gold filename
            output_folder="${folder/answers/evaluation}/$(basename "$model")/"
            output_filename="${folder/answers/evaluation}/$(basename "$model")/${dataset}_test.tsv"
            pred_filename="$folder/$(basename "$model")/${dataset}_test.txt"
            gold_filename="datasets/${dataset}_test.jsonl"

            # File does not exist and folder exists
            if [ ! -e "$output_filename" ] && [ -e "$pred_filename" ]; then
                sbatch sbatch/evaluation_template.sh $output_folder $pred_filename $gold_filename
                echo "$output_filename"
            fi
        done
    done
done

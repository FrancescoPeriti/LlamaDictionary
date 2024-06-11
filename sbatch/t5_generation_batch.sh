#!/bin/bash

datasets=("en-codwe" "oxford" "wordnet" "slang" "wiki" "stop")
models=("flan-t5-definition-en-xl")

# Loop through datasets
for dataset in "${datasets[@]}"/*; do
    if [[ "$dataset" == 'stop/*' ]]; then
        continue
    fi

    for model in "${models[@]}"; do
        # File does not exist
        if [ ! -e "$output_filename" ]; then
            sbatch sbatch/t5_generation_template.sh "datasets/${dataset}_test.jsonl"
        fi
    done
done

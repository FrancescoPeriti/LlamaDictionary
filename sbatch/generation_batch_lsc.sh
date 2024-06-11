#!/bin/bash

#!/bin/bash

main_folders=("dwug_en/wsi" "dwug_en/wsi25" "dwug_en/wsi50" "dwug_en/wsi75" "dwug_en/wsi100")
ft_model_folders=("lora-models/" "lora-models/" "none")
ft_model_names=("llama2chat-1024-2048-0.1-0.001-1e-4" "llama3instruct-512-1024-0.05-0.001-1e-4" "flan-t5-definition-en-xl")
qloras=("False" "False" "none")

for main_folder in "${main_folders[@]}"; do
    echo "$main_folder"
    for filename in "$main_folder"/*; do
        for i in "${!ft_model_folders[@]}"; do
            ft_model_folder="${ft_model_folders[$i]}"
            ft_model_name="${ft_model_names[$i]}"
            qlora="${qloras[$i]}"
            suffix=${main_folder#dwug_en/wsi}

            if [[ "$ft_model_folder" != "lora-models/" ]]; then
                sbatch sbatch/t5_generation_template.sh "$filename" "lsc$suffix-t5-answers"
                continue
            fi

            base_name="$(basename "$filename" .jsonl)"
            new_name="${base_name}.txt"
            output_filename="lsc$suffix-lora-answers/$ft_model_name/$new_name"
            echo "$base_name $filename $new_name $output_filename" "$qlora"
            sbatch sbatch/generation_template.sh $output_filename $model_name "$ft_model_folder/$ft_model_name/final-epoch4" $filename $qlora
        done
    done
done

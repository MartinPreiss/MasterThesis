#!/bin/bash

#nohup /train_all_fusion_models.sh &> train_all_fusion_models.log & disown

# Display current directory
pwd

# Activate pipenv shell
pipenv shell

echo "PID: $!" 

Dataset_Names=("truthfulqa" "haluleval" "refact")

for first_benchmark in "${Dataset_Names[@]}"; do
    for second_benchmark in "${Dataset_Names[@]}"; do
        if [ "$first_benchmark" != "$second_benchmark" ]; then
            echo "Running evaluation for datasets: $first_benchmark and $second_benchmark"
            python -m thesis task=average_in_domain_shift task.first_benchmark=$first_benchmark task.second_benchmark=$second_benchmark model.comparison_method=no_comparison
            python -m thesis task=average_in_domain_shift task.first_benchmark=$first_benchmark task.second_benchmark=$second_benchmark model.freeze_last_layers=False model.comparison_method=no_comparison

        fi
    done 
done

#for first_benchmark in "${Dataset_Names[@]}"; do
#    python -m thesis task=average_in_domain_shift task.first_benchmark=$first_benchmark task.second_benchmark=$first_benchmark task.use_pretrained=False
#done
#!/bin/bash

#nohup /train_all_fusion_models.sh &> train_all_fusion_models.log & disown

# Display current directory
pwd

# Activate pipenv shell
pipenv shell

echo "PID: $!" 

#Dataset_Names=("truthfulqa" "haluleval" "refact")
Dataset_Names=("haluleval")

layer_depth=(3 4 5)
#baseline_name=("last_layer" "middle_layer" "stacked_layers" "all_layers_ensemble")
baseline_name=("last_layer")
for dataset_name in "${Dataset_Names[@]}"; do
    echo "Running evaluation for dataset: $dataset_name"
    for depth in "${layer_depth[@]}"; do
        for baseline_name in "${baseline_name[@]}"; do
            echo "Running evaluation with depth: $depth, baseline name: $baseline_name"
            python -m thesis task=average_earlystopping wandb.use_wandb=False benchmark=$dataset_name model=baseline model.layer_depth=$depth  model.name=$baseline_name
        done           
    done
done
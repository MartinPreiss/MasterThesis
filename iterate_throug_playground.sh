#!/bin/bash

#nohup /train_all_fusion_models.sh &> train_all_fusion_models.log & disown

# Display current directory
pwd

# Activate pipenv shell
pipenv shell

#Dataset_Names=("truthfulqa" "haluleval" "refact")
Dataset_Names=("truthfulqa")
Model_Names=("all_layer_classifier" "layer_fusion_weights" "layer_similarity_classifier" "baseline1" "baseline2" "euclidean_distance")

for dataset_name in "${Dataset_Names[@]}"; do
    #contrastive loss false
    for model in "${Model_Names[@]}"; do
        echo "Running evaluation for model: $model"
        python -m thesis task=playground model=$model 
    done

done
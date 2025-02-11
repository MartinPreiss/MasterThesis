#!/bin/bash

#nohup /train_all_fusion_models.sh &> train_all_fusion_models.log & disown

# Display current directory
pwd

# Activate pipenv shell
pipenv shell

Dataset_Names=("truthfulqa" "haluleval" "refact")

Model_Names=("all_layer_classifier" "gated_layer_fusion" "lstm" "layer_fusion_weights" "layer_similarity_classifier" "ensemble_weight_fusion" "layer_attention" "baseline1" "baseline2" "euclidean_distance")

for dataset_name in "${Dataset_Names[@]}"; do
    echo "Running evaluation for dataset: $dataset_name"
    
    #contrastive loss false
    for model in "${Model_Names[@]}"; do
        echo "Running evaluation for model: $model"
        python -m thesis task=train_layer_fusion wandb.use_wandb=True benchmark=$dataset_name model=$model wandb.name=midterm_ 
    done

done

for dataset_name in "${Dataset_Names[@]}"; do
    echo "Running avg evaluation for dataset: $dataset_name"
    
    #contrastive loss false
    for model in "${Model_Names[@]}"; do
        echo "Running evaluation for model: $model"
        python -m thesis task=average_earlystopping wandb.use_wandb=True benchmark=$dataset_name model=$model wandb.name=midterm_avg 
    done

done
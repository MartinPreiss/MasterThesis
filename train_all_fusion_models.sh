#!/bin/bash

#nohup /train_all_fusion_models.sh &> train_all_fusion_models.log & disown

# Display current directory
pwd

# Activate pipenv shell
pipenv shell

Dataset_Names=("truthfulqa" "haluleval" "refact")

Model_Names=("all_layer_classifier" "layer_fusion" "lstm" "layer_fusion_weights" "layer_similarity_classifier" "ensemble_weight_fusion" "layer_attention")

for dataset_name in "${Dataset_Names[@]}"; do
    echo "Running evaluation for dataset: $dataset_name"
    
    #contrastive loss false
    for model in "${Model_Names[@]}"; do
        echo "Running evaluation for model: $model"
        python -m thesis task=train_layer_fusion wandb.use_wandb=True benchmark=$dataset_name model=$model wandb.name=cf_ task.training_params.use_contrastive_loss=False
        python -m thesis task=train_layer_fusion wandb.use_wandb=True benchmark=$dataset_name model=$model wandb.name=ct_ task.training_params.use_contrastive_loss=True
    done

done

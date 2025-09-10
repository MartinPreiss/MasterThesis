#!/bin/bash

#nohup /train_all_fusion_models.sh &> train_all_fusion_models.log & disown

# Display current directory
pwd

# Activate pipenv shell
pipenv shell

echo "PID: $!" 

#Dataset_Names=("truthfulqa" "haluleval" "refact")
Dataset_Names=("haluleval")

#Model_Names=("layer_similarity_classifier" "baseline1" "baseline2" "euclidean_distance" "all_layer_classifier" "layer_fusion_weights")
#Model_Names=("layer_similarity_classifier" "all_layer_classifier")

#for dataset_name in "${Dataset_Names[@]}"; do
#    echo "Running evaluation for dataset: $dataset_name"
#    
#    #contrastive loss false
#    for model in "${Model_Names[@]}"; do
#        echo "Running evaluation for model: $model"
#        python -m thesis task=train_layer_fusion wandb.use_wandb=True benchmark=$dataset_name model=$model wandb.name=midterm_ 
#    done
#
#done

#for dataset_name in "${Dataset_Names[@]}"; do
#    echo "Running avg evaluation for dataset: $dataset_name"
#    
#    #contrastive loss false
#    for model in "${Model_Names[@]}"; do
#        echo "Running evaluation for model: $model"
#        python -m thesis task=average_earlystopping wandb.use_wandb=True benchmark=$dataset_name model=$model wandb.name=midterm_avg_10 task.training_params.patience=10
#    done
#
#done

layer_depth=(3)
comparison_methods=("no_comparison" "dot_product" "euclidean_norm" "manhatten" "pairwise_dot_product" "euclidean_distance" "manhatten_distance" "cosine")
aggregation_methods=("shared_classifier_ensemble" "flattend_aggregation")
linearity_classifier=(True False)
contrastive_loss=(False)

for dataset_name in "${Dataset_Names[@]}"; do
    echo "Running evaluation for dataset: $dataset_name"
    for depth in "${layer_depth[@]}"; do
        for comparison_method in "${comparison_methods[@]}"; do
            for aggregation_method in "${aggregation_methods[@]}"; do
                for linearity in "${linearity_classifier[@]}"; do
                    for contrastive in "${contrastive_loss[@]}"; do
                        echo "Running evaluation with depth: $depth, comparison method: $comparison_method, aggregation method: $aggregation_method, linearity: $linearity, contrastive loss: $contrastive"
                        python -m thesis task=average_earlystopping wandb.use_wandb=False benchmark=$dataset_name model=layer_comparison_classifier model.layer_depth=$depth model.comparison_method=$comparison_method model.aggregation_method=$aggregation_method model.final_classifier_non_linear=$linearity task.training_params.use_contrastive_loss=$contrastive
                    done
                done
            done
        done
    done
done

#582990
python -m thesis task=train_layer_fusion wandb.use_wandb=True benchmark=haluleval model=layer_comparison_classifier model.comparison_method=no_comparison

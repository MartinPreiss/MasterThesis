#!/bin/bash

#nohup /train_all_fusion_models.sh &> train_all_fusion_models.log & disown

# Display current directory
pwd

# Activate pipenv shell
pipenv shell

echo "PID: $!" 

#Dataset_Names=("truthfulqa" "haluleval" "refact")
Dataset_Names=("refact")

llm_names=(
     "meta-llama/Llama-3.1-8B-Instruct"
     "meta-llama/Llama-3.2-1B-Instruct"
     "meta-llama/Llama-3.2-3B-Instruct"
     "meta-llama/Llama-3.3-70B-Instruct"
     "google/gemma-3-1b-it"
     "google/gemma-3-4b-it"
     "google/gemma-3-12b-it"
     "google/gemma-3-27b-it"
)

#for dataset_name in "${Dataset_Names[@]}"; do
#    echo "Running evaluation for dataset: $dataset_name"
#    for llm_name in "${llm_names[@]}"; do
#        echo "Creating Dataset with LLM: $llm_name"
#        python -m thesis task=create_dataset llm.name=$llm_name llm.model_id=$llm_name benchmark=$dataset_name use_network_mounted=True           
#    done
#done
#4176359 
#31525

for dataset_name in "${Dataset_Names[@]}"; do
   echo "Running evaluation for dataset: $dataset_name"
   for llm_name in "${llm_names[@]}"; do
       echo "Running evaluation with LLM: $llm_name"
       python -m thesis task=average_earlystopping wandb.use_wandb=False benchmark=$dataset_name model=layer_comparison_classifier model.comparison_method=no_comparison llm.name=$llm_name llm.model_id=$llm_name use_network_mounted=True task.training_params.patience=100 task.path_to_save=./thesis/data/different_llms_finetuned task.use_pretrained=False # task.training_params.batch_size=10 task.training_params.learning_rate=0.1
       python -m thesis task=average_earlystopping wandb.use_wandb=False benchmark=$dataset_name model=layer_comparison_classifier model.comparison_method=cosine llm.name=$llm_name llm.model_id=$llm_name use_network_mounted=True task.training_params.patience=100 task.path_to_save=./thesis/data/different_llms_finetuned task.use_pretrained=False # task.training_params.batch_size=10 task.training_params.learning_rate=0.1
                             
    done
done

      
           
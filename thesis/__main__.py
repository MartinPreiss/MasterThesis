
import hydra
from omegaconf import DictConfig, OmegaConf

from thesis.ue.train_per_layer import train_per_layer
from thesis.ue.continue_learning import continue_learning
from thesis.ue.train_log_reg import train_log_reg
from thesis.ue.train_layer_fusion import train_layer_fusion, average_earlystopping
from thesis.data_handling.create_dataset import create_classification_dataset,create_positional_dataset
from thesis.ue.test_on_benchmarks import test_on_benchmarks
from thesis.utils import print_cuda_info
from thesis.baselines.baseline import evaluate_baseline

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    options = {
        "train_per_layer": train_per_layer,
        "continue_learning":continue_learning,
        "train_log_reg": train_log_reg,
        "train_layer_fusion":train_layer_fusion,
        "create_dataset": create_classification_dataset,
        "create_positional_dataset": create_positional_dataset,
        "test_on_benchmarks": test_on_benchmarks,
        "average_earlystopping": average_earlystopping,
        "evaluate_baseline": evaluate_baseline,
        "playground": playground
    }

    print_cuda_info()
    options[cfg.task.name](cfg)

def playground(cfg):
    from thesis.baselines.baseline import SelfCheckGPT
    #perform small check
    model_id = "google/gemma-2-9b-it"
    question = "What is the capital of France?"
    answer = "The capital of France is Berlin."
    scg = SelfCheckGPT()
    scg.detect(model_id,question,answer)
    
    """
    from thesis.xai.pca_analysis import main as pca_main,pca_per_layer
    pca_per_layer(cfg)
    #get_statistics_of_early_stopping_runs(cfg)
    """
    """
    from thesis.utils import print_number_of_parameters
    from thesis.models.model_handling import get_model
    
    embedding_size = 3584  # first batch, first input #embedding size
    num_layers = 42  # first batch, first input #embedding size
    print("Embedding Size", embedding_size, "Number of Layers", num_layers)
    
    model = get_model(cfg,embedding_size=embedding_size, num_layers=num_layers)
    
    print(cfg.model.name)
    print_number_of_parameters(model)
    """

def get_statistics_of_early_stopping_runs(cfg):
    
    import os
    import torch
    import glob
    import numpy as np
    
    path = "./thesis/data/avgs_early_stopping"
    files = glob.glob(f"{path}/*.pth")

    for file in files:
        #file_format = baseline2_refact__5f1s
        file_name = os.path.basename(file)
        
        if not "f1" in file_name:
            continue
        print(os.path.basename(file))
        metric = file_name.split("_")[-1]
        benchmark_name = file_name.split("_")[-2]
        f1_scores = torch.load(file)
        
        # Extract model name and benchmark name from the file name
        #filename = os.path.basename(file)
        #model_name, benchmark_name, _ = filename.split('__')[0].split('_')

        print(f"File: {file_name}")
        print(f"Min F1: {f1_scores.min().item()}")
        print(f"Max F1: {f1_scores.max().item()}")
        print(f"Median F1: {f1_scores.median().item()}")
        print(f"Avg F1 (excluding zeros): {f1_scores[f1_scores > 0].mean().item()}")
        print()
    

if __name__ == "__main__":
    main()
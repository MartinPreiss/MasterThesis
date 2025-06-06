
import hydra
from omegaconf import DictConfig, OmegaConf

from thesis.ue.train_per_layer import train_per_layer
from thesis.ue.continue_learning import continue_learning
from thesis.ue.train_log_reg import train_log_reg
from thesis.ue.train_layer_fusion import train_layer_fusion, average_earlystopping, average_in_domain_shift
from thesis.ue.train_positional_layer_fusion import train_positional_layer_fusion, average_positional_runs
from thesis.data_handling.create_dataset import create_classification_dataset,create_positional_dataset
from thesis.ue.test_on_benchmarks import test_on_benchmarks
from thesis.utils import print_cuda_info
#from thesis.baselines.baseline import evaluate_baseline

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
        "average_in_domain_shift": average_in_domain_shift,
        "train_positional_layer_fusion": train_positional_layer_fusion,
        "average_positional_runs": average_positional_runs,
        #"evaluate_baseline": evaluate_baseline,
        "playground": playground
    }
    print(cfg)
    print_cuda_info()
    options[cfg.task.name](cfg)

def playground(cfg):
    from thesis.data_handling.data_handling import get_positional_dataset, get_dataloaders
    from thesis.data_handling.locate import convert_tagging2onehot
    import torch
    from tqdm import tqdm

    dataset = get_positional_dataset(cfg)
    cfg.task.positional_task = True
    cfg.task.training_params.batch_size = 1
    data_loaders = get_dataloaders(cfg, dataset)
    # get label distribution
    ratios = []
    num_tokens = []
    total_positives = 0
    for dataloader, name in zip(data_loaders, ["train", "val", "test"]):
        data_loader_positives = 0
        data_loader_num_tokens = 0
        for batch in tqdm(dataloader):
            samples, labels = batch
            labels = convert_tagging2onehot(labels).flatten()
            # get label distribution
            data_loader_positives += labels.sum().item()
            #print(labels.shape)
            data_loader_num_tokens += len(labels)
        ratios.append((data_loader_positives / data_loader_num_tokens)* 100)
        num_tokens.append(data_loader_num_tokens)
        total_positives += data_loader_positives
    print(f"Train ratio: {ratios[0]:.2f}")
    print(f"Val ratio: {ratios[1]:.2f}")
    print(f"Test ratio: {ratios[2]:.2f}")
    print(f"num_tokens {num_tokens[0]} {num_tokens[1]} {num_tokens[2]}")
    print(f"total num tokens: {num_tokens[0] + num_tokens[1] + num_tokens[2]}")
    print(f"total positives: {total_positives}")
    print(f"total ratio: {(total_positives / (num_tokens[0] + num_tokens[1] + num_tokens[2]))* 100:.2f}")


#    # get label distribution per benchmark 
#    from thesis.data_handling.data_handling import get_embedding_dataset, get_dataloaders
#    import torch
#    benchmark_names = ["refact","haluleval","truthfulqa"]
#    llms = ["meta-llama/Llama-3.1-8B-Instruct" ,"google/gemma-3-1b-it" ,"google/gemma-3-4b-it" ,"google/gemma-3-12b-it" ,"google/gemma-3-27b-it" ,"meta-llama/Llama-3.2-1B-Instruct" ,"meta-llama/Llama-3.2-3B-Instruct" ,"meta-llama/Llama-3.3-70B-Instruct"]
#   
#    latex_table = """\\begin{table}[h]
#\\centering
#\\begin{tabular}{l cc ccc}
#\\hline
#\\textbf{Benchmark} & \\textbf{num\_llm\_layers} & \\textbf{embedding\_size} & \\multicolumn{3}{c}{\\textbf{Ratio of Positives}} \\\\
#                   &                      &                      & \\textbf{Train} & \\textbf{Val} & \\textbf{Test} \\\\
#\\hline"""
#    for model_name in llms: 
#        cfg.benchmark.name = "truthfulqa"
#        cfg.llm.name = model_name
#        dataset = get_embedding_dataset(cfg)
#        data_loaders = get_dataloaders(cfg, dataset)
#        
#        samples, labels = next(iter(data_loaders[0]))
#        embedding_size = samples.shape[-1]
#        num_llm_layers = samples.shape[-2]
#        
#        num_positives = 0
#        # get label distribution
#        ratios = []
#        for dataloader, name in zip(data_loaders, ["train", "val", "test"]):
#            data_loader_positives = 0
#            for batch in dataloader:
#                samples, labels = batch
#                # get label distribution
#                data_loader_positives += labels.sum().item()
#            num_positives += data_loader_positives
#            ratios.append(data_loader_positives / len(dataloader.dataset)* 100) 
#        latex_table += f"""{model_name[model_name.rfind("/")+1:]} & {num_llm_layers} & {embedding_size} & {ratios[0]:.2f} & {ratios[1]:.2f} & {ratios[2]:.2f} \\\\ \\hline \n"""
#    latex_table += """\\hline
#\\end{tabular}
#\\caption{Dataset LLM layer count, embedding size, and positive label ratios for each benchmark and split.}
#\\label{tab:dataset_ratios}
#\\end{table}"""
#    print(latex_table)







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
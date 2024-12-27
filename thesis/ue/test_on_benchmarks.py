# import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from thesis.models.model_handling import get_model
from thesis.metrics import calculate_metrics
from thesis.data_handling.data_handling import get_embedding_dataset, get_dataloaders
from thesis.utils import print_number_of_parameters, get_device, init_wandb

from thesis.models.loss.contrastive_loss import ContrastiveLoss
from omegaconf import DictConfig
import warnings
import wandb
import datetime

from tqdm import tqdm
import os
import yaml

warnings.filterwarnings("always")

device = get_device()

def get_benchmark_names()->list:
    path_to_configs = "thesis/config/benchmark"
    
    benchmark_names = []
    for file_name in os.listdir(path_to_configs):
        if file_name.endswith(".yaml"):
            with open(os.path.join(path_to_configs, file_name), 'r') as file:
                config = yaml.safe_load(file)
                benchmark_names.append(config.get('name'))

    return benchmark_names
    
    
def evaluate_model(cfg: DictConfig, data_loader,model,benchmark_name):
    classification_loss = nn.BCEWithLogitsLoss().to(device)
    contrastive_loss = ContrastiveLoss().to(device)
    all_preds = []
    all_labels = []
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            val_outputs, encoded_space = model(inputs, cfg.task.training_params.use_contrastive_loss)
            if cfg.task.training_params.use_contrastive_loss:
                val_loss += classification_loss(val_outputs, labels) + contrastive_loss(encoded_space, labels)
            else:
                val_loss += classification_loss(val_outputs, labels)
            all_preds.append(val_outputs)
            all_labels.append(labels)
    val_loss /= len(data_loader)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_preds = torch.where(all_preds >= 0.0, 1.0, 0.0)
    acc, prec, rec, f1 = calculate_metrics(preds=all_preds, labels=all_labels)
    
    return val_loss, acc, prec, rec, f1
    
def test_on_benchmarks(cfg: DictConfig):
    print("STILL TESTING ONLY ON VALIDATION")
    cfg.wandb.name = "test_" + str(cfg.wandb.name).replace("None","")
    init_wandb(cfg)
    
    # Create a table for bar chart
    table = wandb.Table(columns=["Benchmark", "Loss","Accuracy", "Precision", "Recall", "F1 Score"])
    
    benchmark_names = get_benchmark_names()
    model_path = cfg.task.model_path
    
    first_benchmark = benchmark_names[0]
    
    
    for benchmark_name in benchmark_names:
        cfg.benchmark.name = benchmark_name 
        # Load the dataset
        dataset = get_embedding_dataset(cfg)
        train_loader, val_loader, test_loader = get_dataloaders(cfg, dataset)
        embedding_size = dataset[0][0].shape[-1]
        num_layers = dataset[0][0].shape[-2]

        if benchmark_name == first_benchmark:
            #dont load model for every benchmark
            model = get_model(cfg, embedding_size=embedding_size, num_layers=num_layers).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

        val_loss, acc, prec, rec, f1 = evaluate_model(cfg, data_loader=val_loader, model=model,benchmark_name=benchmark_name)

        

        table.add_data(benchmark_name,val_loss, acc, prec, rec, f1)
    
    # Log the table to wandb
    wandb.log({f"metrics_table": table})
    
    # Create bar charts for each metric
    metrics = ["Loss", "Accuracy", "Precision", "Recall", "F1 Score"]
    for metric in metrics:
        wandb.log({f"{metric}_bar_chart": wandb.plot.bar(table, "Benchmark", metric, title=f"{metric}")})

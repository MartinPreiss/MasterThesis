import pandas as pd
import gc
import torch 
import os
import pickle
import wandb

def get_absolute_path(relative_path:str):
    basePath = os.path.dirname(os.path.abspath(__file__))
    return basePath + relative_path

def print_cuda_info():

    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    if torch.cuda.is_available():
        print("CUDA is available")
    print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])

def write_pickle(data,filename,path):
    with open(path+filename + ".pkl", "wb") as file:
        pickle.dump(data, file)

def load_pickle(filename,path):
    with open(path+filename + ".pkl", "rb") as file:
        my_list = pickle.load(file)
    return my_list

def clear_all__gpu_memory():
    # Delete all variables that reference tensors on the GPU
    for obj in dir():
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            del obj



    # Empty the CUDA cache
    torch.cuda.empty_cache()
    
    # Call garbage collector to free up memory
    gc.collect()

    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def clear_unused_gpu_memory():
    # Call garbage collector to free up memory
    gc.collect()

    # Empty the CUDA cache
    torch.cuda.empty_cache()
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
def print_number_of_parameters(model):

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_wandb(cfg):
    if cfg.wandb.use_wandb:
        
        default_name = cfg.model.name 
        wandb.init(
            project=cfg.wandb.project_name + "_" + cfg.benchmark.name if cfg.wandb.project_name != "None" else None,
            name=  cfg.wandb.name + default_name   if cfg.wandb.name else default_name,
            entity = "martinpreiss",
            group=cfg.wandb.group_name if cfg.wandb.group_name != "None" else None,
            config=dict(cfg)) 
import torch 
import os

def print_cuda_info():

    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(torch.cuda.is_available())
    print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
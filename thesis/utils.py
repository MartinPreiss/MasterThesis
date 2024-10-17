import pandas as pd
import gc
import torch 
import os
import pickle

from thesis.config import REFACT_PATH
from thesis.prompts import system_prompt_valid_invalid, user_prompt_valid_invalid, simple_prompt

def get_gemma_prompt(cot=False):
    if not cot: 
        prompt = system_prompt_valid_invalid.split("Evaluation Process:")[0].strip() + user_prompt_valid_invalid.split("Thought process:")[0].strip()
        print(prompt)
    else:
        prompt =system_prompt_valid_invalid + user_prompt_valid_invalid
        
    return prompt 

def get_df(tag_type=None):
    path= REFACT_PATH
    df = pd.read_json(path, lines=True, orient="records")
    if tag_type:
        df = df[df["tag_type"] == tag_type]
    df = df.dropna()
    return df

def get_prompt_df():
    df = get_df()
    prompt = simple_prompt
    df["original_prompt"] = df.apply(lambda row: prompt.format(question=row["question"],answer =row["answer"]), axis=1 )
    df["transformed_prompt"]  = df.apply(lambda row: prompt.format(question=row["question"],answer =row["transformed_answer"]), axis=1)
    return df[["transformed_prompt","original_prompt"]]

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
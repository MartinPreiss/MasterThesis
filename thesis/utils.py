import pandas as pd
import torch 
import os

def get_df(tag_type=None):
    path= "/home/knowledgeconflict/home/martin/hpi-mp-facts-matter/data/full_download.jsonl"
    df = pd.read_json(path, lines=True, orient="records")
    df = df[df["is_correct"] == True]
    if tag_type:
        df = df[df["tag_type"] == tag_type]
    df = df.dropna()
    return df

prompt = """Question: {question}

Answer: {answer}

Does the answer contain only factually correct statements? Only answer with Yes or No!
"""

def get_prompt_df():
    df = get_df()
    
    df["original_prompt"] = df.apply(lambda row: prompt.format(question=row["question"],answer =row["answer"]), axis=1 )
    df["transformed_prompt"]  = df.apply(lambda row: prompt.format(question=row["question"],answer =row["transformed_answer"]), axis=1)
    return df[["transformed_prompt","original_prompt"]]

def print_cuda_info():

    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    if torch.cuda.is_available():
        print("CUDA is available")
    print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
        
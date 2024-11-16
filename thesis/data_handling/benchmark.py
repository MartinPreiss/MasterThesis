import pandas as pd 
from thesis.data_handling.prompts import system_prompt_valid_invalid, user_prompt_valid_invalid, simple_prompt
from thesis.utils import get_absolute_path
def get_gemma_prompt(cot=False):
    if not cot: 
        prompt = system_prompt_valid_invalid.split("Evaluation Process:")[0].strip() + user_prompt_valid_invalid.split("Thought process:")[0].strip()
        print(prompt)
    else:
        prompt =system_prompt_valid_invalid + user_prompt_valid_invalid
        
    return prompt 

def get_df(cfg):
    if cfg.benchmark.name == "refact":
        return get_refact_df(cfg)
    elif cfg.benchmark.name == "truthfulqa":
        return get_truthfulqa_df(cfg)
    
    raise(Exception("Dataset not found with name:",cfg.benchmark.name ))
    
def get_truthfulqa_df(cfg):
    raise(Exception("not implemented"))
    
    
def get_refact_df(cfg):
    path= get_absolute_path(cfg.benchmark.path)
    df = pd.read_json(str(path), lines=True, orient="records")
    if cfg.benchmark.tag_type:
        df = df[df["tag_type"] == cfg.benchmark.tag_type]
    df = df.dropna()
    return df

def get_prompt(cfg):
    if cfg.benchmark.prompt_name == "simple_prompt":
        return simple_prompt
    raise(Exception("Prompt Name for Benchmark not found"))

def get_prompt_df(cfg):
    
    df = get_df(cfg)
    prompt = get_prompt(cfg)
    df["original_prompt"] = df.apply(lambda row: prompt.format(question=row["question"],answer =row["answer"]), axis=1 )
    df["transformed_prompt"]  = df.apply(lambda row: prompt.format(question=row["question"],answer =row["transformed_answer"]), axis=1)
    return df[["transformed_prompt","original_prompt"]]
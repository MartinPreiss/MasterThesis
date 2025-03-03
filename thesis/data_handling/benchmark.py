import pandas as pd 
from thesis.data_handling.prompts import system_prompt_valid_invalid, user_prompt_valid_invalid, simple_prompt, qa_format
from thesis.utils import get_absolute_path
def get_gemma_prompt(cot=False):
    if not cot: 
        prompt = system_prompt_valid_invalid.split("Evaluation Process:")[0].strip() + user_prompt_valid_invalid.split("Thought process:")[0].strip()
        print(prompt)
    else:
        prompt =system_prompt_valid_invalid + user_prompt_valid_invalid
        
    return prompt 

def get_df(cfg):
    #needs to be columns=[question,answer,fake_answer]
    if cfg.benchmark.name == "refact":
        return get_refact_df(cfg)
    elif cfg.benchmark.name == "truthfulqa":
        return get_truthfulqa_df(cfg)
    elif cfg.benchmark.name == "haluleval":
        return get_haluleval_df(cfg)
    raise(Exception("Dataset not found with name:",cfg.benchmark.name ))
    
def get_truthfulqa_df(cfg):
    path= get_absolute_path(cfg.benchmark.path)
    df = pd.read_csv(path)
    df = df[['Question', 'Best Answer','Incorrect Answers']]
    
    df = df.rename(columns={
    "Question": "question",
    "Best Answer": "answer",
    "Incorrect Answers": "fake_answer"})
    
    df["fake_answer"] = df.apply(lambda row: row["fake_answer"][:row["fake_answer"].find(";")], axis=1 )
    
    return df
    
def get_haluleval_df(cfg):
    path= get_absolute_path(cfg.benchmark.path)
    df = pd.read_json(str(path), lines=True, orient="records")
    
    df = df[['question', 'right_answer','hallucinated_answer']]
    
    df = df.rename(columns={
    "question": "question",
    "right_answer": "answer",
    "hallucinated_answer": "fake_answer"})
    
    
    return df   
    
    
def get_refact_df(cfg):
    path= get_absolute_path(cfg.benchmark.path)
    df = pd.read_json(str(path), lines=True, orient="records")
    if cfg.benchmark.tag_type:
        df = df[df["tag_type"] == cfg.benchmark.tag_type]
    df = df.dropna()
    
    df = df.rename(columns={
    "transformed_answer": "fake_answer",
    "transformed_answer_tagged": "fake_answer_tagged",
    })
    return df

def get_prompt(cfg):
    if cfg.task.prompt_name == "simple_prompt":
        return simple_prompt
    raise(Exception("Prompt Name for Benchmark not found"))

def get_prompt_df(cfg):
    
    df = get_df(cfg)
    prompt = get_prompt(cfg)
    df["original_prompt"] = df.apply(lambda row: prompt.format(question=row["question"],answer =row["answer"]), axis=1 )
    df["transformed_prompt"]  = df.apply(lambda row: prompt.format(question=row["question"],answer =row["fake_answer"]), axis=1)
    return df[["transformed_prompt","original_prompt"]]

def get_qa_df(cfg):
    df = get_df(cfg)
    df["original_qa"] = df.apply(lambda row: qa_format.format(question=row["question"],answer =row["answer"]), axis=1 )
    df["transformed_qa"]  = df.apply(lambda row: qa_format.format(question=row["question"],answer =row["fake_answer"]), axis=1)
    df["tagged_transformed_qa"]  = df.apply(lambda row: qa_format.format(question=row["question"],answer =row["fake_answer_tagged"]), axis=1)
    return df[["transformed_qa","original_qa","tagged_transformed_qa"]]
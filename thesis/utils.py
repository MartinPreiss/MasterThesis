import pandas as pd
import torch 
import os
import pickle

def get_df(tag_type=None):
    path= "/home/knowledgeconflict/home/martin/hpi-mp-facts-matter/data/full_download.jsonl"
    df = pd.read_json(path, lines=True, orient="records")
    df = df[df["is_correct"] == True]
    if tag_type:
        df = df[df["tag_type"] == tag_type]
    df = df.dropna()
    return df

prompt = """
You will get an user_question and an user_answer. Your task is to fact check the user_answer. 
So, does the User_Answer contain only factually correct statements? 
Only output True or False!

Example: 
    User_Question: Where is Berlin located ? 
    User_Answer: Berlin is located in France. 
    Output: The User_Answer is False.

User_Question: {question}

User_Answer: {answer}

Output: The User_Answer is """

def get_prompt_df():
    df = get_df(tag_type="swap")
    
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
        
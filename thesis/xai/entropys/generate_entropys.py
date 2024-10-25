
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_token_pos_of_substring(text,substring,tokenizer):

    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    tokenized_string = ' '.join(tokens)
    
    # Tokenize the substring
    substring_tokens = tokenizer.tokenize(substring)
    tokenized_substring = ' '.join(substring_tokens)
    
    # Find the index of the tokenized substring in the tokenized string
    start_idx = tokenized_string.find(tokenized_substring)
    
    # Calculate the token index
    if start_idx != -1:
        # Number of spaces before the tokenized substring in the tokenized string
        space_count = tokenized_string[:start_idx].count(' ')
        substring_token_start_idx = space_count
        substring_token_end_idx = space_count + len(substring_tokens) - 1
        print(f"Token index range for '{substring}' is: ({substring_token_start_idx}, {substring_token_end_idx})")
    else:
        print(f"Substring '{substring}' not found in the tokenized text.")
        print(tokens)
        print(substring_tokens)
        return None,None
    
    


    return substring_token_start_idx, substring_token_end_idx

prompt = """Question: {question}

Answer: {answer}

Does the answer contain only factually correct statements?
"""

def calculate_attention_entropy(prompt,model,tokenizer):

    input_ids = tokenizer(prompt, return_tensors="pt")
    input_id_list = input_ids["input_ids"][0].tolist()# Batch index 0
    input_ids =input_ids.to('cuda')
    output = model.generate(**input_ids,max_length=len(input_id_list)*2, return_dict_in_generate=True, output_scores=True, output_attentions=True)

    attention = output["attentions"]
    sequences = output["sequences"]
    tokens = tokenizer.convert_ids_to_tokens(input_id_list) 

    # Convert the predicted token IDs to text
    decoded_text = tokenizer.decode(sequences[0])


    #preprocess
    attention = attention[0] #only first generation


    attention = torch.stack(tuple(layer[:,:,:,:len(tokens)] for layer in attention))

    # Calculate entropy along the last dimension
    normalized_entropy = -torch.sum(attention * torch.log(attention + 1e-9), dim=-1) / torch.log(torch.tensor(len(tokens)))

    return normalized_entropy, decoded_text

def get_attention(prompt,model,tokenizer):

    input_ids = tokenizer(prompt, return_tensors="pt")
    input_id_list = input_ids["input_ids"][0].tolist()# Batch index 0
    input_ids =input_ids.to('cuda')
    output = model.generate(**input_ids,max_length=len(input_id_list)*2, return_dict_in_generate=True, output_scores=True, output_attentions=True)

    attention = output["attentions"]
    sequences = output["sequences"]
    tokens = tokenizer.convert_ids_to_tokens(input_id_list) 

    # Convert the predicted token IDs to text
    decoded_text = tokenizer.decode(sequences[0])


    #preprocess
    attention = attention[0] #only first generation


    attention = torch.stack(tuple(layer[:,:,:,:len(tokens)] for layer in attention))

    return attention, decoded_text

def get_df():
    path= "/home/knowledgeconflict/home/martin/hpi-mp-facts-matter/data/full_download_test.jsonl"
    df = pd.read_json(path, lines=True, orient="records")
    df = df[df["is_correct"] == True]
    df = df[df["tag_type"] == "swap"]
    df = df.dropna()
    df[df['intermediate_results'].apply(lambda x: "entities" in x.keys())]
    df[df['intermediate_results'].apply(lambda x: len(x["entities"]) == 1)]
    df[df['transformed_answer_tagged'].apply(lambda x: x.count("<swap>") == 1)]
    return df

def get_token_input_ids(prompt,tokenizer):

    input_ids = tokenizer(prompt, return_tensors="pt")
    input_id_list = input_ids["input_ids"][0].tolist()# Batch index 0
    input_ids =input_ids.to('cuda')
    tokens = tokenizer.convert_ids_to_tokens(input_id_list) 

    return tokens, input_ids


def main():

    df = get_df()

    model_id = 'google/gemma-2-27b-it'
    tokenizer = AutoTokenizer.from_pretrained(model_id,device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True,device_map="auto")

 
    df["original_output"] = ""
    df["transformed_output"] = ""
    df["original_first_token_pos"] = np.nan
    df["transformed_first_token_pos"]= np.nan
    df["original_last_token_pos"]= np.nan
    df["transformed_last_token_pos"]= np.nan


    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        #original
        original_dir_path = "./data/prompt_entropys/original/"
        original_prompt = prompt.format(question=row["question"],answer =row["answer"]) 
        original_entropy, original_decoded_text = calculate_attention_entropy(prompt=original_prompt,model=model,tokenizer=tokenizer)
        original_entity = row['intermediate_results']["entities"][0]
        original_first_token_pos, original_last_token_pos  = find_token_pos_of_substring(original_prompt,original_entity,tokenizer)

        #transformed
        transformed_dir_path = "./data/prompt_entropys/transformed/"
        transformed_prompt = prompt.format(question=row["question"],answer =row["transformed_answer"]) 
        transformed_entropy, transformed_decoded_text = calculate_attention_entropy(prompt=transformed_prompt,model=model,tokenizer=tokenizer)
        transformed_entity = row['intermediate_results']["new_entities"][0]
        transformed_first_token_pos, transformed_last_token_pos  = find_token_pos_of_substring(transformed_prompt,transformed_entity,tokenizer)

        
        
        torch.save(original_entropy, original_dir_path + row["id"] + '.pt')
        torch.save(transformed_entropy, transformed_dir_path + row["id"] + '.pt')

        df.loc[index, "original_output"] = original_decoded_text
        df.loc[index, "transformed_output"] = transformed_decoded_text
        df.loc[index, "original_first_token_pos"] = original_first_token_pos
        df.loc[index, "transformed_first_token_pos"]= transformed_first_token_pos
        df.loc[index, "original_last_token_pos"]=  original_last_token_pos
        df.loc[index, "transformed_last_token_pos"]= transformed_last_token_pos

    df.to_json("./data/prompt_entropys/data.json", lines=True, orient="records")



    

if __name__ == '__main__':
    main()
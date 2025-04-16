import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from thesis.data_handling.benchmark import get_prompt_df, get_qa_df
from thesis.data_handling.tag_helper import parse_tags

from thesis.xai.hook import get_layer_hooks

from torch.utils.data import TensorDataset, ConcatDataset


from tqdm import tqdm

def get_last_pos_embeddings(hooks):
    embeddings = []
    for hook in hooks:
        embeddings.append(hook.data[:,-1,:].cpu())  # postprocess (hook only saves last token )     
        hook.clear_hook()
        
    return torch.cat(embeddings)

def get_all_pos_embeddings(hooks):
    embeddings = []
    for hook in hooks:
        embeddings.append(hook.data.cpu())  # postprocess (hook only saves last token )     
        hook.clear_hook()
        
    return torch.cat(embeddings)

def get_embedding_dataset(prompts,model,tokenizer, hooks,true_output:str,wrong_output:str):
    
    sample_embeddings = []
    labels = []
        
    for index, prompt in tqdm(prompts.items()):
        input_ids = tokenizer(prompt, return_tensors="pt").to('cuda')
        #forward pass and final output
        with torch.no_grad():
            output = model.generate(**input_ids,max_length=len(input_ids[0])+1, return_dict_in_generate=True, output_scores=True)

        # Convert the predicted token IDs to text
        decoded_text = tokenizer.decode(output["sequences"][0][-1])
        
        #get_embedding
        embeddings = get_last_pos_embeddings(hooks)
        
        #get label
        if decoded_text == true_output:
            label = 0
        elif decoded_text == wrong_output:
            label = 1
        else:
            print("FAILED for decoded text",decoded_text)
            continue
        
        labels.append(label)
        sample_embeddings.append(embeddings)
    
    return TensorDataset(torch.stack(sample_embeddings),torch.Tensor(labels).unsqueeze(1))

def save_positional_embeddings(qa_df,model,tokenizer,hooks,dataset_path):
    
    #qa_df[["transformed_qa","original_qa","tagged_transformed_qa"]]
    for index, row in tqdm(qa_df.iterrows()):
        
        #encode original_qa only (no_output)
        input_ids = tokenizer(row["original_qa"], return_tensors="pt").to('cuda')
        with torch.no_grad():
            model(**input_ids)
        
        #get_embedding
        internal_embeddings = get_all_pos_embeddings(hooks)
    
        #get label
        labels = torch.zeros_like(input_ids["input_ids"]) #all correct
        torch.save(internal_embeddings,f"{dataset_path}/embeddings_{index}.pth")
        torch.save(labels,f"{dataset_path}/labels_{index}.pth")
        
        #encode original_qa only (no_output)
        input_ids = tokenizer(row["transformed_qa"], return_tensors="pt").to('cuda')
        with torch.no_grad():
            model(**input_ids)
        
        #get_embedding
        internal_embeddings = get_all_pos_embeddings(hooks)
        
        #parse tags of untokenized_text
        plain_text, tag_info = parse_tags(row["tagged_transformed_qa"])
        #tag_info = [((11, 14), 'swap'), ((16, 30), 'neg')] of untokenized text
        
        #map tag_info to tokenized text
        tokenized_tag_info = []
        for (start,end),tag in tag_info:
            start = tokenizer(row["transformed_qa"], return_offsets_mapping=True).char_to_token(start)
            end = tokenizer(row["transformed_qa"], return_offsets_mapping=True).char_to_token(end)
            tokenized_tag_info.append(((start,end),tag))	
        
        #get labels with tokenized_tag_info
        labels = torch.zeros_like(input_ids["input_ids"])
        for (start,end),tag in tokenized_tag_info:
            labels[0,start:end] = 1
            
        
        #quality assurance (compare tokens with labels)
        tokens = tokenizer.convert_ids_to_tokens(input_ids["input_ids"][0])
        assert len(tokens) == len(labels[0])
        positive_tokens_list = [tokens[i] for i in range(len(tokens)) if labels[0,i] == 1]
        positive_tokens = " ".join(positive_tokens_list)
        #save positive_tokens to file
        with open(f"{dataset_path}/positive_tokens_{index}.txt","w") as f:
            f.write(positive_tokens)
        
        torch.save(internal_embeddings,f"{dataset_path}/embeddings_{index+len(qa_df)}.pth")
        torch.save(labels,f"{dataset_path}/labels_{index+len(qa_df)}.pth")
   
def create_classification_dataset(cfg):
    
    df = get_prompt_df(cfg)
    df_original_prompt = df["original_prompt"]
    df_fake_prompt = df["transformed_prompt"]
    
    model_id = cfg.llm.name
    tokenizer = AutoTokenizer.from_pretrained(model_id,device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True,device_map="auto")
    print(model)
    print(len(model.model.layers))
    
    #hook layers
    hooks = get_layer_hooks(model)
    
    original_dataset = get_embedding_dataset(df_original_prompt,model,tokenizer,hooks,true_output="TRUE",wrong_output="FALSE")
    transformed_dataset = get_embedding_dataset(df_fake_prompt,model,tokenizer,hooks,true_output="FALSE",wrong_output="TRUE")
    
    combined_dataset = ConcatDataset([original_dataset, transformed_dataset])
    model_name = model_id[model_id.rfind("/")+1:]
    torch.save(combined_dataset,f"/mnt/vast-gorilla/martin.preiss/datasets/last_token/embedding_{model_name}_{cfg.benchmark.name}.pth")

def create_positional_dataset(cfg):
    
    # get refact 
    df = get_qa_df(cfg)
    
    # get model
    model_id = cfg.llm.name
    tokenizer = AutoTokenizer.from_pretrained(model_id,device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True,device_map="auto")
    print(model)
    print(len(model.model.layers))
    
    #hook layers
    hooks = get_layer_hooks(model)
    
    #prepeare dataset_path (create folder if not there already)
    model_name = model_id[model_id.rfind("/")+1:]
    
    dataset_path = f"/mnt/vast-gorilla/martin.preiss/datasets/positions/embedding_{model_name}_{cfg.benchmark.name}/"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=False)
    
    save_positional_embeddings(df,model,tokenizer,hooks,dataset_path)
    

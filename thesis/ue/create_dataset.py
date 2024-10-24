import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from thesis.utils import get_prompt_df, print_cuda_info,clear_unused_gpu_memory, clear_all__gpu_memory,write_pickle

from thesis.xai.logit_lens import get_hooks,get_label_ids

from torch.utils.data import TensorDataset, ConcatDataset


from tqdm import tqdm

def get_embeddings(hooks):
    embeddings = []
    for hook in hooks:
        embeddings.append(hook.data[:,-1,:].cpu())  # postprocess (hook only saves last token )     
        hook.clear_hook()
        
    return torch.cat(embeddings)

def get_embedding_dataset(prompts,model,tokenizer, hooks,true_output:str,wrong_output:str):
    #true_ids, false_ids = get_label_ids(tokenizer)
    
    sample_embeddings = []
    labels = []
        
    for index, prompt in tqdm(prompts.items()):
        input_ids = tokenizer(prompt, return_tensors="pt").to('cuda')
        #forward pass and final output
        with torch.no_grad():
            output = model.generate(**input_ids,max_length=len(input_ids[0])+1, return_dict_in_generate=True, output_scores=True,temperature=0.1)

        # Convert the predicted token IDs to text
        decoded_text = tokenizer.decode(output["sequences"][0][-1])
        
        #get_embedding
        embeddings = get_embeddings(hooks)
        
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
if __name__ == "__main__":    
    
    df = get_prompt_df()
    df_original_prompt = df["original_prompt"]
    df_fake_prompt = df["transformed_prompt"]
    
    #model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model_id = 'google/gemma-2-27b-it'
    tokenizer = AutoTokenizer.from_pretrained(model_id,device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True,device_map="auto")
    print(model)
    print(len(model.model.layers))
    
    #hook layers
    hooks = get_hooks(model)
    
    original_dataset = get_embedding_dataset(df_original_prompt,model,tokenizer,hooks,true_output="TRUE",wrong_output="FALSE")
    transformed_dataset = get_embedding_dataset(df_fake_prompt,model,tokenizer,hooks,true_output="FALSE",wrong_output="TRUE")
    
    combined_dataset = ConcatDataset([original_dataset, transformed_dataset])
    model_name = model_id[model_id.rfind("/")+1:]
    torch.save(combined_dataset,f"./data/datasets/embeddings/embedding_{model_name}_all.pth")

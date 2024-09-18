import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from thesis.utils import get_prompt_df, print_cuda_info

from thesis.xai.hook import Hook

def logit_lens(model):
    #returns logit_map for every layer 
    
    #hook every layer and save output
        
    #last mlp layer 
        
    #return logits 
    pass



if __name__ == "__main__":    
    
    print_cuda_info()
    
    model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id,device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True,device_map="auto")

    print(model)
    
    df = get_prompt_df()[:1]
    
    original_prompt = df["original_prompt"].item()
    fake_prompt = df["transformed_prompt"].item()
    
    input_ids = tokenizer(fake_prompt, return_tensors="pt")
    input_id_list = input_ids["input_ids"][0].tolist()# Batch index 0
    input_ids = input_ids.to('cuda')
    
    #hook layer
    hooks = []
    for layer in model.model.layers: 
        hooks.append(Hook(layer))
    
    #forward pass and final output
    output = model.generate(**input_ids,max_length=len(input_id_list)+1, return_dict_in_generate=True, output_scores=True)
    sequences = output["sequences"]
    print(sequences)
    # Convert the predicted token IDs to text
    decoded_text = tokenizer.decode(sequences[0])
    
    print("original",decoded_text)
    #calculate logit_lens
    layer_outputs = []
    for hook in hooks:
        
        layer_outputs.append(model.lm_head(hook.data))
    
    for id, layer_output in enumerate(layer_outputs):
        layer_ids = torch.argmax(layer_output,dim=-1)
        print(layer_ids.shape)
        print(id,tokenizer.decode(layer_ids[0][-1]))
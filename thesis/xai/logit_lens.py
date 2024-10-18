import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from thesis.utils import get_prompt_df, print_cuda_info,clear_unused_gpu_memory, clear_all__gpu_memory,write_pickle

from thesis.xai.hook import Hook
from tqdm import tqdm


def logit_lens(model, hooks):
    layer_outputs = []
    with torch.no_grad():
        for hook in hooks:
            layer_output = model.lm_head(model.model.norm(hook.data)) #logit lens
            hook.clear_hook()
            layer_output = layer_output[:,-1,:] # postprocess (hook only saves last token )
            #layer_output = torch.nn.functional.softmax(layer_output,dim=1)
            layer_outputs.append(layer_output)  
    return layer_outputs

def get_hooks(model,layer_ids=None):
    if not layer_ids:
        layer_ids = range(len(model.model.layers))
    hooks = []
    for id,layer in enumerate(model.model.layers): 
        if id in layer_ids:
            hooks.append(Hook(layer))

    return hooks

def get_label_ids(tokenizer):    
    true_words = ["true","True","TRUE"]
    false_words = ["false", "False", "FALSE"]

    # Tokenize and get token IDs for each word in the list
    true_ids = [tokenizer.encode(word, add_special_tokens=False) for word in true_words]
    true_ids = torch.tensor(true_ids).flatten().to('cuda')#.cpu()
    false_ids = [tokenizer.encode(word, add_special_tokens=False) for word in false_words]
    false_ids = torch.tensor(false_ids).flatten().to('cuda')#.cpu()
    
    return true_ids, false_ids

def save_layer_outputs(prompts,model,tokenizer, hooks,output_path):
    true_ids, false_ids = get_label_ids(tokenizer)
    
    original_answers = []
    labeled_true_scores = []
    labeled_false_scores = []
    topk_tokens = []
    topk_scores = []
    
    for index, prompt in tqdm(prompts.items()):
        input_ids = tokenizer(prompt, return_tensors="pt").to('cuda')
        #forward pass and final output
        with torch.no_grad():
            output = model.generate(**input_ids,max_length=len(input_ids[0])+1, return_dict_in_generate=True, output_scores=True,temperature=0.1)

        # Convert the predicted token IDs to text
        decoded_text = tokenizer.decode(output["sequences"][0][-1])

        original_answers.append(decoded_text)
        
        #calculate logit_lens / get intermediate scores
        layer_scores = logit_lens(model,hooks)
        #calculate metric 
        layer_labeled_true_scores = []
        layer_labeled_false_scores = []
        layer_topk_scores = []
        layer_topk_tokens = []
        
        for id, layer_output in enumerate(layer_scores):
            #layer_output = torch.nn.functional.relu(layer_output) #only add positive 
            labeled_true_score = torch.sum(torch.index_select(layer_output,dim=-1,index=true_ids)).reshape(1)
            labeled_false_score = torch.sum(torch.index_select(layer_output,dim=-1,index=false_ids)).reshape(1)
            #print(labeled_true_score.item()-labeled_false_score.item())

            topk = layer_output.topk(10,dim=-1)
            topk_layer_ids = topk[1]
            topk_score = topk[0]
            topk_token = tokenizer.decode(topk_layer_ids[0][-1])
            
            layer_labeled_true_scores.append(labeled_true_score)
            layer_labeled_false_scores.append(labeled_false_score)
            layer_topk_scores.append(topk_score)
            layer_topk_tokens.append(topk_token)
        
        labeled_true_scores.append(torch.cat(layer_labeled_true_scores).cpu())
        labeled_false_scores.append(torch.cat(layer_labeled_false_scores).cpu())
        topk_scores.append(torch.stack(layer_topk_scores).cpu())
        topk_tokens.append(layer_topk_tokens)
        
        del input_ids, output, layer_scores#
        clear_all__gpu_memory()
        

        #print(torch.cuda.memory_summary())

    write_pickle(original_answers,filename="original_answers",path=output_path)
    torch.save(torch.stack(labeled_true_scores),output_path + "labeled_true_scores" + ".pt")
    torch.save(torch.stack(labeled_false_scores),output_path + "labeled_false_scores" + ".pt")
    write_pickle(topk_tokens,filename="topk_tokens",path=output_path)
    torch.save(torch.stack(topk_scores),output_path + "topk_scores" + ".pt")


if __name__ == "__main__":    
    
    print_cuda_info()
    
    df = get_prompt_df()
    
    df_original_prompt = df["original_prompt"]
    df_fake_prompt = df["transformed_prompt"]

    #model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model_id = 'google/gemma-2-9b-it'
    tokenizer = AutoTokenizer.from_pretrained(model_id,device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True,device_map="auto")
    
    print(model)
    
    
    #hook layers
    hooks = get_hooks(model)

    save_layer_outputs(df_original_prompt,model,tokenizer,hooks,output_path="/home/knowledgeconflict/home/martin/MasterThesis/data/logit_lens/original/")
    save_layer_outputs(df_fake_prompt,model,tokenizer,hooks,output_path="/home/knowledgeconflict/home/martin/MasterThesis/data/logit_lens/fake/")

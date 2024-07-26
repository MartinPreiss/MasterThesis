
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import pandas as pd

def find_first_token_id_of_substring(text,substring,tokenizer):

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
    print(substring_token_start_idx)


    return substring_token_start_idx, substring_token_end_idx


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
    entropy = -torch.sum(attention * torch.log(attention + 1e-9), dim=-1)

    return entropy, decoded_text

def get_df():
    path= ""
    df = pd.read_json()

def main():
    model_id = 'google/gemma-2-27b-it'
    name = "original_token"
    prompt = "Issac Asimov is a woman. Is this true? Only Yes or No!"

    model_id = 'google/gemma-2-27b-it'
    tokenizer = AutoTokenizer.from_pretrained(model_id,device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True,device_map="auto")
    calculate_attention_entropy(prompt=prompt,substring="woman",model=model,tokenizer=tokenizer,name=name)

    first_fake_token_id, _  = find_first_token_id_of_substring(prompt,substring,tokenizer)


if __name__ == '__main__':
    main()
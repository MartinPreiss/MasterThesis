from bertviz import head_view, model_view
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


name = "original_token_2"
prompt = "Issac Asimov is a man. Is this true? Only Yes or No!"

model_id = 'google/gemma-2-27b-it'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True)
print(model)
input_ids = tokenizer(prompt, return_tensors="pt")
input_id_list = input_ids["input_ids"][0].tolist() # Batch index 0
output = model.generate(**input_ids,max_length=len(input_id_list)*2, return_dict_in_generate=True, output_scores=True, output_attentions=True)

attention = output["attentions"]
sequences = output["sequences"]
tokens = tokenizer.convert_ids_to_tokens(input_id_list) 

# Convert the predicted token IDs to text
decoded_text = tokenizer.decode(sequences[0])

print("Decoded text:", decoded_text)
print("len tokens",len(tokens))
print("dim attention attention",len(attention))
print("len first indice attention",len(attention[0]))

#preprocess
attention = attention[0] #only first generation
attention = tuple(layer[:,:,:,:len(tokens)] for layer in attention)

"""
html_page = head_view(attention, tokens, html_action="return")

with open("head_view.html", "w") as file:
    file.write(html_page.data)
    

html_page = model_view(attention, tokens, html_action="return")

with open("model_view.html", "w") as file:
    file.write(html_page.data)
"""

#only_first_fake token
# Convert tuple to list to allow modifications
attention_list = list(attention)
first_fake_token_id = 7
entropys = []
for layer_id, layer in enumerate(attention_list):
    for batch_id, batch in enumerate(layer):
        
        head_entropys = []
        for head_id, head in enumerate(batch):
            for token_id, token_attention in enumerate(head):
                if token_id != first_fake_token_id:
                    attention_list[layer_id][batch_id][head_id][token_id]  = torch.zeros(token_attention.shape)
                if token_id == first_fake_token_id:
                    #normalized_attention = token_attention / torch.sum(token_attention)
                    entropy = -torch.sum(token_attention * torch.log(token_attention + 1e-9))
                    head_entropys.append(float(entropy))
        entropys.append(head_entropys)        

df = pd.DataFrame(entropys)

# Print the DataFrame to verify
print(df)

# Step 4: Save the DataFrame to a CSV file (optional)
df.to_csv(name+'_weight_entropys.csv', index=False)

"""
attention = tuple(attention_list)
html_page = model_view(attention, tokens, html_action="return")

# Create the histogram
plt.hist(entropys, bins=5, edgecolor='black')

# Add titles and labels
plt.title('Histogram of Sample Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.savefig(name+".png")

with open(name+"_view.html", "w") as file:
    file.write(html_page.data)
"""
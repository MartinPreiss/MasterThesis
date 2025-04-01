## methods to achieve sequence labeling 
import torch
import numpy as np

def convert_onehot2bio(tensor): 
    
   #get 1 ranges as tuples 
    one_indices = torch.nonzero(tensor).flatten()
    
    #print(one_indices)
    # where numbers dont neighbor each other by 1
    
    #fill with values
    bio_values = torch.ones(one_indices.shape[-1])
    
    diff = one_indices[:-1] - one_indices[1:]   
    bio_values[0] = 1
    for i in range(len(diff)):
        if diff[i] == -1:
            bio_values[i+1] = 2
        else:
            bio_values[i+1] = 1
    
    
    bio_tensor = torch.zeros(tensor.shape)
    bio_tensor[one_indices] = bio_values
                
    return bio_tensor

def convert_onehot2bioes(tensor):
    
    #get 1 ranges as tuples 
    one_indices = torch.nonzero(tensor).flatten()
    
    #fill with values
    # 0 is o
    # 1 is b
    # 2 is i
    # 3 is e
    # 4 is s
    
    
    breaks = torch.where(torch.diff(one_indices) != 1,True,False)  
    start = 0
    tuples = []
    for i in range(len(breaks)):
        if breaks[i]:
            tuples.append((start,i))
            start = i+1
            tuples.append((start,len(breaks)))
        else: 
            continue
    
    bio_values = torch.ones(one_indices.shape[-1])
    for tuple in tuples: 
        start  = tuple[0]
        end = tuple[1]
        if start == end:
            bio_values[start] = 4
        else: 
            bio_values[start] = 1
            bio_values[end] = 3
            bio_values[start+1:end] = 2
    
    bio_tensor = torch.zeros(tensor.shape)
    bio_tensor[one_indices] = bio_values
    
    return bio_tensor

    

def convert_onehot2tagging_scheme(tensor,tag_scheme="BIO"):
    #check wether tensor is only filled with 0,1s 
    
    #check wether tensor is not only of length 1 or 0
    #assert tensor.shape[-1] != 1 or tensor.shape[-1] != 0
    
    if tag_scheme == "IO":
        return tensor
    elif tag_scheme == "BIO":
        return convert_onehot2bio(tensor)
    elif tag_scheme == "BIOES":
        return convert_onehot2bioes(tensor)
    
def convert_tagging2onehot(tensor):
    non_zero_indices = torch.nonzero(tensor)
    
    tensor[non_zero_indices] = 1
    
    return tensor
    
    
if __name__ == "__main__":
    tensors = [torch.tensor([0, 1, 0]), torch.tensor([1, 0,1,1,1,1, 0]), torch.tensor([0, 0, 1,1])]
    for tensor in tensors:
        print(convert_onehot2tagging_scheme(tensor,"BIOES"))
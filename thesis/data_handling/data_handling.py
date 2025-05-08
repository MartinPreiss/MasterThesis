import torch 
import os
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from thesis.utils import get_device
from thesis.data_handling.benchmark import get_df
import re

from thesis.data_handling.locate import convert_onehot2tagging_scheme

from torch.nn.utils.rnn import pad_sequence

def train_collate_fn(batch):
    # Separate inputs and labels
    inputs, labels = zip(*batch)
    """
    # Pad inputs to the same length
    inputs_padded = pad_sequence(inputs, batch_first=True)
    # Stack labels (assuming labels are already of the same size)
    labels_padded = pad_sequence(labels, batch_first=True)
    """
    inputs_concat = torch.cat(inputs, dim=0)
    labels_concat = torch.cat(labels, dim=0).squeeze()
    return inputs_concat, labels_concat


def get_embedding_dataset(cfg):
    model_name = cfg.llm.name[cfg.llm.name.rfind("/")+1:]
    if cfg.use_network_mounted:
        dataset_path = f"/mnt/vast-gorilla/martin.preiss/datasets/last_token/"
    else:
        dataset_path = f"/home/martin.preiss/MasterThesis/thesis/data/datasets/embeddings/"
    print("loading dataset from",dataset_path)

    dataset = torch.load(f"{dataset_path}embedding_{model_name}_{cfg.benchmark.name}.pth",weights_only=False)#,map_location=torch.device('cpu'))   
    if cfg.pca.use_pca:
        raise(Exception("WARNING: PCA not only performed on test set")) 
        dataset = PCADataset(dataset,n_components=cfg.pca.n_components,layer_wise=cfg.pca.layer_wise)
    if  cfg.use_coveig:
        dataset = CovEigDataset(dataset)
        
    return dataset

def get_positional_dataset(cfg): 
    model_name = cfg.llm.name[cfg.llm.name.rfind("/")+1:]
    dataset_path = f"/mnt/vast-gorilla/martin.preiss/datasets/positions/embedding_{model_name}_{cfg.benchmark.name}/"
    print("loading dataset from",dataset_path)

    dataset = PositionalDataset(dataset_path,cfg.task.tag_scheme)

    return dataset

def get_refact_split(cfg,test_val_size):
    #get original id
    df = get_df(cfg)    
    df["id"] = df.apply(lambda row: row["unique_row_id"][:row["unique_row_id"].find(".")],axis=1)
    
    #get_unique_ids
    grouped = df.groupby("id").indices
    unique_ids = list(grouped.keys())
    
    #problem
    #len(X)=2002, len(df)=1001

    #perform split on unique_ids
    train_ids, temp_ids = train_test_split(
        unique_ids, train_size=test_val_size, random_state=cfg.seed
    ) 
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=cfg.seed
    )
    
    # Collect indices for each set based on the `id`s
    train_indices = [i for id_ in train_ids for i in grouped[id_]]
    val_indices = [i for id_ in val_ids for i in grouped[id_]]
    test_indices = [i for id_ in test_ids for i in grouped[id_]]
    
    # Return split indices and corresponding `Y` data
    return (train_indices, val_indices, test_indices)

def print_label_data(y, split_type="total"):
    
    num_positive = (y == 1).sum().item()
    num_negative =  len(y) - num_positive
    print(f"Percentage of {split_type} positive samples {num_positive / len(y):.2f}")
    print(f"Percentage of {split_type} negative samples {num_negative / len(y):.2f}")

def perform_train_val_test_split(cfg,dataset):
        
    # Split the dataset
    data_size = len(dataset)
    train_ratio = 0.7
    train_size = int(0.7 * data_size)
    test_val_size = data_size - train_size 
    x_indices = list(range(data_size))
    #split indices (sklearn cant handle shapes greater than 3 :) ) 
    
    len_first_half = len(x_indices)//2
    if cfg.benchmark.name == "refact":
        train_indices, val_indices, test_indices = get_refact_split(cfg,test_val_size)
    
    else:
        #first half
        train_indices, temp_indices = train_test_split(
            x_indices[:len_first_half], train_size=train_ratio, stratify=None, random_state=cfg.seed
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, stratify=None, random_state=cfg.seed
        )
    #get other half
        #get other half
    train_indices.extend([id+len_first_half for id in train_indices])
    val_indices.extend([id+len_first_half for id in val_indices])
    test_indices.extend([id+len_first_half for id in test_indices])
    

    # Create subsets for train, val, and test
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Print some statistics
    print("Dataset Size:", data_size)
    print("Trainset Size:", len(train_dataset))
    print("Valset Size:", len(val_dataset))
    print("Testset Size:", len(test_dataset))

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(cfg,dataset):
    
    train_dataset, val_dataset, test_dataset = perform_train_val_test_split(cfg,dataset)

    # Create DataLoaders for training and validation
    batch_size = cfg.task.training_params.batch_size
    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=False, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False,num_workers=0, pin_memory=False, collate_fn=train_collate_fn)
    test_loader = DataLoader(test_dataset,batch_size=1, shuffle=False,num_workers=0, pin_memory=False)
    
    return train_loader, val_loader, test_loader

def get_dataloader_for_layer(data_loader,layer_id,batch_size):
    
    
        all_data = torch.cat([data[0][:, layer_id, :] for data in data_loader],dim=0)
        labels = torch.cat([data[1] for data in data_loader],dim=0)
        
        return DataLoader(TensorDataset(all_data,labels),batch_size=batch_size)


class PCADataset(Dataset):
    def __init__(self, dataset, n_components=100,layer_wise=False):
        
        # Stack all data to fit the PCA
        all_data = torch.stack([dataset[i][0] for i in range(len(dataset))])
        old_shape = all_data.shape
        #PCA expects X as shape (num_samples,n_features)
        
        #PCA
        pca = PCA(n_components=n_components)
        if layer_wise:
            num_layers = dataset[0][0].shape[-2] 
            all_data = all_data.cpu().numpy()
            transformed_data = torch.stack([torch.Tensor(pca.fit_transform(all_data[:,layer_id,:])) for layer_id in range(num_layers)]).permute(1, 0, 2)
        else:
            all_data = all_data.view(old_shape[0]*old_shape[1],-1).cpu().numpy()
            transformed_data = pca.fit_transform(all_data)
    
        self.data = torch.tensor(transformed_data, dtype=torch.float32).squeeze().reshape((old_shape[0],old_shape[1],transformed_data.shape[-1]))
        self.labels = torch.stack([dataset[i][1] for i in range(len(dataset))])    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx], self.labels[idx]
    
class CovEigDataset(Dataset):
    def __init__(self, dataset):
                
        coveigs = []
        labels = []
        
        for sample, label in tqdm(dataset):
            
            cov = torch.cov(sample)
            #cov = torch.transpose(sample)#* jd * sample#Σ = Z⊤ · Jd · Z
            
            eigenvalues, _ = torch.linalg.eig(cov)
            
            coveigs.append(eigenvalues)
            labels.append(label)
    
        self.data = torch.stack(coveigs)
        self.labels = torch.stack(labels)    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx], self.labels[idx]    

class PositionalDataset(Dataset):
    def __init__(self, path, tag_scheme):
        self.dataset_path = path
        self.tag_scheme = tag_scheme
    
    def __len__(self):
        # count number of files in the dataset directory with the prefix "embedding_"
        number_of_files = len([f for f in os.listdir(self.dataset_path) if f.startswith("embeddings_")])
        print("Number of files in dataset:", number_of_files)
        return number_of_files 
    def __getitem__(self, idx):
        pos_embeddings = torch.load(f"{self.dataset_path}embeddings_{idx}.pth")
        labels = convert_onehot2tagging_scheme(torch.load(f"{self.dataset_path}labels_{idx}.pth"),tag_scheme=self.tag_scheme)
        return pos_embeddings, labels 
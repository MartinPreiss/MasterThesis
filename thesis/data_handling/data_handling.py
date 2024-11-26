import torch 
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from thesis.utils import get_device
from thesis.data_handling.benchmark import get_df


def get_embedding_dataset(cfg):
    model_name = cfg.llm.name[cfg.llm.name.rfind("/")+1:]
    dataset = torch.load(f"thesis/data/datasets/embeddings/embedding_{model_name}_{cfg.benchmark.name}.pth")#,map_location=torch.device('cpu'))   
    if cfg.pca.use_pca:
        raise(Exception("WARNING: PCA not only performed on test set")) 
        dataset = PCADataset(dataset,n_components=cfg.pca.n_components,layer_wise=cfg.pca.layer_wise)
    if  cfg.use_coveig:
        dataset = CovEigDataset(dataset)
        
    return dataset

def get_refact_split(cfg,Y,test_val_size):
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
    

def get_dataloaders(cfg,dataset):
    
    #prepare for dataset_spliting
    X = torch.stack([dataset[i][0] for i in range(len(dataset))])
    Y = torch.stack([dataset[i][1] for i in range(len(dataset))])
    
    # Split the dataset
    data_size = len(dataset)
    train_size = int(0.7 * data_size)
    test_val_size = data_size - train_size 
    x_indices = list(range(X.shape[0]))
    #split indices (sklearn cant handle shapes greater than 3 :) ) 
    
    len_first_half = len(x_indices)//2
    if cfg.benchmark.name == "refact":
        x_indices = get_refact_split(cfg,Y,test_val_size)
        train_indices, val_indices, test_indices = x_indices
    else:
        #first half
        indices_first_half = x_indices[:len_first_half]
        Y_first_half = Y[:len_first_half]
        train_indices, X_temp, _, y_temp = train_test_split(indices_first_half, Y_first_half, test_size=test_val_size, stratify=Y_first_half, random_state=cfg.seed)
        val_indices, test_indices, _, _ = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=cfg.seed)
        
    #get other half
        #get other half
    train_indices.extend([id+len_first_half for id in train_indices])
    val_indices.extend([id+len_first_half for id in val_indices])
    test_indices.extend([id+len_first_half for id in test_indices])
    
    #make tensors 
    train_indices = torch.tensor(train_indices)
    val_indices = torch.tensor(val_indices)
    test_indices = torch.tensor(test_indices)
    
    #get y_data
    y_train = torch.index_select(Y, 0, train_indices)
    y_val = torch.index_select(Y, 0, val_indices)
    y_test = torch.index_select(Y, 0, test_indices)


    #create torch_datasets
    train_dataset = TensorDataset(torch.index_select(X.cpu(), 0, train_indices.cpu()),y_train)
    val_dataset = TensorDataset(torch.index_select(X.cpu(), 0, val_indices.cpu()),y_val)
    test_dataset = TensorDataset(torch.index_select(X.cpu(), 0, test_indices.cpu()),y_test)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset,batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=100, shuffle=False)
    
    #print some statistics 
    print("Dataset Size", data_size)
    print("Trainset Size", train_size)
    print("Valset Size", len(val_dataset))
    print("Testset Size", len(test_dataset))
    print_label_data(Y)
    print_label_data(y_train,"y_train")
    print_label_data(y_val,"y_val")
    print_label_data(y_test,"y_test")
    return train_loader, val_loader, test_loader


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
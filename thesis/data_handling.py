import torch 
from torch.utils.data import Dataset,DataLoader, TensorDataset
from torch.utils.data import random_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def get_embedding_dataset(cfg):
    dataset = torch.load(f"/home/knowledgeconflict/home/martin/MasterThesis/data/datasets/embeddings/embedding_{cfg.llm_model_name}_all.pth")   
    if cfg.pca.use_pca:    
        dataset = PCADataset(dataset,n_components=cfg.pca.n_components,layer_wise=cfg.pca.layer_wise)
    return dataset

def get_dataloaders(cfg,dataset):
    
    #prepare for dataset_spliting
    
    
    X = torch.stack([dataset[i][0] for i in range(len(dataset))]).cpu().numpy()
    Y = torch.stack([dataset[i][0] for i in range(len(dataset))]).cpu().numpy()
    # Split the dataset
    data_size = len(dataset)
    train_size = int(0.7 * data_size)
    test_val_size = data_size - train_size 

    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=test_val_size, stratify=Y, random_state=cfg.seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=cfg.seed)

    #create torch_datasets
    train_dataset = TensorDataset(torch.Tensor(X_train),torch.Tensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val),torch.Tensor(y_val))
    test_dataset = TensorDataset(torch.Tensor(X_test),torch.Tensor(y_test))

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset,batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=100, shuffle=False)
        
    print("Dataset Size", data_size)
    print("Trainset Size", train_size)
    print("Valset Size", len(val_dataset))
    print("Valset Size", len(test_dataset))
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
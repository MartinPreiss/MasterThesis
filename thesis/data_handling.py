import torch 
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import random_split
from sklearn.decomposition import PCA

def get_embedding_dataset(cfg):
    dataset = torch.load(f"./data/datasets/embeddings/embedding_{cfg.llm_model_name}_all.pth")   
    if cfg.pca.use_pca:    
        dataset = PCADataset(dataset,n_components=cfg.pca.n_components,layer_wise=cfg.pca.layer_wise)
    return dataset

def get_dataloaders(dataset):

    # Split the dataset
    data_size = len(dataset)
    train_size = int(0.8 * data_size)
    val_size = data_size - train_size 
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("Dataset Size", data_size)
    print("Trainset Size", train_size)
    print("Valset Size", val_size)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset,batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=100, shuffle=False)
    
    return train_loader, val_loader


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
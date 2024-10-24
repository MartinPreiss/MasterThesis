import torch 
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def get_embedding_dataset(model_name):
    return torch.load(f"./data/datasets/embeddings/embedding_{model_name}_all.pth")   

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
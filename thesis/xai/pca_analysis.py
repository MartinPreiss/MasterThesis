
from thesis.data_handling.data_handling import get_embedding_dataset, perform_train_val_test_split
from thesis.utils import  init_wandb
from omegaconf import DictConfig

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import wandb

def transform_dataset_to_numpy(dataset,last_layer_only=False):
    
    # Stack all data to fit the PCA
    X = torch.stack([dataset[i][0] for i in range(len(dataset))])
    Y = torch.stack([dataset[i][1] for i in range(len(dataset))]).cpu().numpy()   
    old_shape = X.shape
    
    if last_layer_only:
        X = X[:,-1,:].squeeze().cpu().numpy()
    else:
        X = X.view(old_shape[0]*old_shape[1],-1).cpu().numpy()   
        # have to broadcast the labels to the same shape as X
        Y = np.repeat(Y, old_shape[1], axis=0) 

    print("Shape of X",X.shape)
    print("Shape of Y",Y.shape)
    return X,Y

def transform_dataset_to_numpy_per_layer(dataset,layer_id):
    
    # Stack all data to fit the PCA
    X = torch.stack([dataset[i][0][layer_id] for i in range(len(dataset))]).cpu().numpy()
    
    return X
    

def main(cfg : DictConfig):
    
    init_wandb(cfg)
    
    # Load the dataset
    dataset = get_embedding_dataset(cfg)
    
    # dataset = Subset(dataset,range(10))
    dataset, _, _ = perform_train_val_test_split(cfg,dataset)

    embedding_size = dataset[0][0].shape[-1]  # first batch, first input #embedding size
    num_layers = dataset[0][0].shape[-2]  # first batch, first input #embedding size
    print("Embedding Size", embedding_size, "Number of Layers", num_layers)
    
    #transform dataset into numpy stuff 
    
    X,Y = transform_dataset_to_numpy(dataset,last_layer_only=True)
    
    # Compute PCA
    pca = PCA()
    pca.fit(X)
    
    # Get explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Compute cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Print or plot
    print("Explained variance per component:", explained_variance_ratio)
    print("Cumulative explained variance:", cumulative_variance)
    
    
    #compare different hidden_size values 
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance vs. Hidden Size")
    plt.grid()

    # Log plot to wandb
    wandb.log({"PCA Explained Variance vs. Hidden Size": wandb.Image(plt)})
    
    #choose an optimal hidden_size 
    target_variance = 0.95
    optimal_hidden_size = np.argmax(cumulative_variance >= target_variance)
    
    print(f"Optimal hidden size for {target_variance*100}% variance: {optimal_hidden_size}")
    
    # per layer plot 

def pca_per_layer(cfg : DictConfig):
    cfg.wandb.name = "PCA Per Layer"
    init_wandb(cfg)
    
    # Load the dataset
    dataset = get_embedding_dataset(cfg)
    
    # dataset = Subset(dataset,range(10))
    dataset, _, _ = perform_train_val_test_split(cfg,dataset)

    embedding_size = dataset[0][0].shape[-1]  # first batch, first input #embedding size
    num_layers = dataset[0][0].shape[-2]  # first batch, first input #embedding size
    print("Embedding Size", embedding_size, "Number of Layers", num_layers)
    
    optimal_hidden_sizes = []
    for layer_id in range(num_layers):
        
        X = transform_dataset_to_numpy_per_layer(dataset,layer_id)
        pca = PCA()
        pca.fit(X)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio) 
        
        target_variance = 0.95
        optimal_hidden_size = np.argmax(cumulative_variance >= target_variance)
        
        optimal_hidden_sizes.append(optimal_hidden_size)
    
    #create plot layer dimension as x and optimal hidden size as y
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(optimal_hidden_sizes) + 1), optimal_hidden_sizes, marker='o', linestyle='--')
    plt.xlabel("Layer")
    plt.ylabel("Principal Components")
    plt.title("PCA Components per Layer with target variance 0.95") 
    plt.grid()
    wandb.log({"Optimal Hidden Size per Layer": wandb.Image(plt)})
    
    optimal_hidden_sizes = np.array(optimal_hidden_sizes)
    print("Avg optimal hidden size",np.mean(optimal_hidden_sizes))
    wandb.log({"Avg optimal hidden size":np.mean(optimal_hidden_sizes)})
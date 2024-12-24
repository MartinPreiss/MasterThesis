# import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from thesis.models.model_handling import get_model
from thesis.metrics import calculate_metrics
from thesis.data_handling.data_handling import get_embedding_dataset, get_dataloaders
from thesis.utils import print_number_of_parameters, get_device, init_wandb

from thesis.models.loss.contrastive_loss import ContrastiveLoss
from omegaconf import DictConfig
import warnings
import wandb

from tqdm import tqdm

warnings.filterwarnings("always")

device = get_device()


def train(cfg,model, train_loader, val_loader):
    
    
    # Loss and optimizer
    if cfg.task.training_params.use_cross_entropy_weighting:    
        train_labels = torch.cat([y for x,y in train_loader])
        num_positive = (train_labels == 1).sum().item()
        num_negative =  len(train_labels) - num_positive
        pos_weight =torch.ones([1]) * (num_negative / num_positive)
    else:
        pos_weight = torch.ones([1])    
        
    print("Using Train Weight: ",pos_weight)
    classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    print("Use Contrastive Loss: ",cfg.task.training_params.use_contrastive_loss)
    contrastive_loss = ContrastiveLoss().to(device)
    print("Use Adam Optimizer with LR: ",cfg.task.training_params.learning_rate, "Weight Decay: ",cfg.task.training_params.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=cfg.task.training_params.learning_rate,weight_decay=cfg.task.training_params.weight_decay)
    print("Train on epochs: ",cfg.task.training_params.epochs)
    epochs = cfg.task.training_params.epochs 
    
    contrast_loss = torch.Tensor([0]).to(device)

    max_f1 = 0
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = (
                data  # Assuming your dataset returns input features and labels# Move inputs and labels to the device (CPU or GPU)
            )
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs , encoded_space = model(inputs,cfg.task.training_params.use_contrastive_loss)
            class_loss = classification_loss(outputs, labels)
            if cfg.task.training_params.use_contrastive_loss:
                contrast_loss = contrastive_loss(encoded_space,labels)
                        
            loss =  class_loss + contrast_loss
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Log the loss
            running_loss += loss.item()
        
        if cfg.wandb.use_wandb:
            wandb.log({"train_loss":running_loss,"classification_loss":class_loss.item(),"contrastive_loss":contrast_loss.item()})
            
        # validation
        all_preds = []
        all_labels = []
        val_loss = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                val_outputs,encoded_space = model(inputs,cfg.task.training_params.use_contrastive_loss)
                if cfg.task.training_params.use_contrastive_loss:
                    val_loss = classification_loss(val_outputs, labels) + contrastive_loss(encoded_space,labels)
                else:
                    val_loss = classification_loss(val_outputs, labels)
                val_loss += loss.item()
                all_preds.append(val_outputs)
                all_labels.append(labels)
        val_loss /= len(val_loader)
        # Calculate metrics after each epoch
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_preds = torch.where(
            all_preds >= 0.0, 1.0, 0.0
        )  # Convert logits to binary predictions
        acc, prec, rec, f1 = calculate_metrics(
            preds=all_preds, labels=all_labels
        )
        running_loss /= len(train_loader)
        
        max_f1 = f1 if f1 > max_f1 else max_f1
        if cfg.wandb.use_wandb:
            wandb.log(
            data={
                "val_loss":val_loss,
                "val_acc": acc,
                "val_precision": prec,
                "val_recall": rec,
                "f1": f1,
            }
        )
    if cfg.wandb.use_wandb:
        wandb.log({"max_f1":max_f1})
        
    # Save the model checkpoint
    # torch.save(model.state_dict(), "simple_classifier.pth")

def train_layer_fusion(cfg : DictConfig):
    
    init_wandb(cfg)
    # Load the dataset
    dataset = get_embedding_dataset(cfg)
    
    # dataset = Subset(dataset,range(10))
    train_loader, val_loader, test_loader = get_dataloaders(cfg,dataset)

    embedding_size = dataset[0][0].shape[-1]  # first batch, first input #embedding size
    num_layers = dataset[0][0].shape[-2]  # first batch, first input #embedding size
    print("Embedding Size", embedding_size, "Number of Layers", num_layers)

    model = get_model(cfg,embedding_size=embedding_size,num_layers=num_layers).to(
        device
    )

    print_number_of_parameters(model)

    train(cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    model.plot_classifier_weights()

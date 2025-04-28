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
import datetime

from tqdm import tqdm
import os

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
    min_val_loss = torch.inf
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
                    batch_val_loss = classification_loss(val_outputs, labels) + contrastive_loss(encoded_space,labels)
                else:
                    batch_val_loss = classification_loss(val_outputs, labels)
                val_loss += batch_val_loss.item()
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
        
        if f1 > max_f1:
            max_f1 = f1 
            checkpoint_model = model.state_dict()
        
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
            
        if cfg.task.training_params.early_stopping:
            if val_loss < min_val_loss:
                min_val_loss = val_loss 
                early_stopping_checkpoint = model.state_dict()
                early_stopping_counter = 0
            else:
                if early_stopping_counter >= cfg.task.training_params.patience:
                    print("Early Stopping")
                    break
                early_stopping_counter += 1

    if cfg.wandb.use_wandb:
        wandb.log({"max_f1":max_f1})
    
    
    # get final evaluation
    if cfg.task.training_params.early_stopping:
        model.load_state_dict(early_stopping_checkpoint)
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
                val_loss += val_loss.item()
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
        if cfg.wandb.use_wandb:
            wandb.log(
                data={
                    "ckpt_loss":val_loss,
                    "ckpt_acc": acc,
                    "ckpt_precision": prec,
                    "ckpt_recall": rec,
                    "ckpt_f1": f1,
                    "early_stopping_epoch":epoch
                }
            )
        
    # Save the model checkpoint
    if cfg.task.training_params.save_model:
        date = datetime.datetime.now().strftime("%H_%M__%d_%m_%Y")
        model_path = f"thesis/data/models/{cfg.model.name}_{cfg.benchmark.name}_{date}.pth"
        print("saving model to ",model_path)
        if cfg.task.training_params.early_stopping:
            torch.save(early_stopping_checkpoint,model_path)
        else:
            torch.save(checkpoint_model,model_path)

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
    
    if cfg.task.use_pretrained: 
        model.load_state_dict(torch.load(cfg.model.pretrained_model_path))
        model.freeze_last_layers()

    print_number_of_parameters(model)

    train(cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    try:    
        model.plot_classifier_weights()
    except:
        print("Could not plot classifier weights")

def finetune_layer_fusion(cfg : DictConfig):
    
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
    
    
    try:    
        model.plot_classifier_weights()
    except:
        print("Could not plot classifier weights")

def get_validation_metrics(cfg,model,val_loader):
    all_preds = []
    all_labels = []
    val_loss = 0
    for i, data in enumerate(val_loader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            val_outputs,encoded_space = model(inputs,cfg.task.training_params.use_contrastive_loss)
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
    
    return acc, prec, rec, f1

def average_earlystopping(cfg : DictConfig):
    
    init_wandb(cfg)

    
    
    path= "./thesis/data/avgs_early_stopping"
    benchmark_name = cfg.benchmark.name
    model_name = cfg.model.name
    contrastive_loss = cfg.task.training_params.use_contrastive_loss

    if cfg.model.name == "layer_comparison_classifier":
        # load important parameters out of config 
        num_classes = cfg.model.num_classes
        comparison_method = cfg.model.comparison_method
        aggregation_method = cfg.model.aggregation_method
        final_classifier_non_linear = cfg.model.final_classifier_non_linear
        layer_depth = cfg.model.layer_depth

        file_name = f"{model_name}_{benchmark_name}__{num_classes}_{comparison_method}_{aggregation_method}_{final_classifier_non_linear}_{layer_depth}_{contrastive_loss}_{cfg.task.training_params.patience}"
    
    else:
        file_name = f"{model_name}_{benchmark_name}_{cfg.task.training_params.patience}"
    
    if os.path.exists(f"{path}/{file_name}_f1s.pth"): 
        print("File already exists, skipping")
        return

    # Load the dataset
    dataset = get_embedding_dataset(cfg)
    
    # dataset = Subset(dataset,range(10))
    train_loader, val_loader, test_loader = get_dataloaders(cfg,dataset)

    embedding_size = dataset[0][0].shape[-1]  # first batch, first input #embedding size
    num_layers = dataset[0][0].shape[-2]  # first batch, first input #embedding size
    print("Embedding Size", embedding_size, "Number of Layers", num_layers)
    
    cfg.wandb.use_wandb = False
    accs,precs, recs, f1s = [],[],[],[]
    for i in tqdm(range(cfg.task.number_of_runs)):
        model = get_model(cfg,embedding_size=embedding_size,num_layers=num_layers).to(
        device
        )
        train(cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        acc, prec, rec, f1 = get_validation_metrics(cfg,model,test_loader)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    
    # calculate average and log for wandb 
    accs = torch.Tensor(accs)
    precs = torch.Tensor(precs)
    recs = torch.Tensor(recs)
    f1s = torch.Tensor(f1s)
    if cfg.wandb.use_wandb:
        wandb.log(
            data={
                "avg_acc": accs.mean(),
                "avg_precision": precs.mean(),
                "avg_recall": recs.mean(),
                "avg_f1": f1s.mean(),
                "std_acc": accs.std(),
                "std_precision": precs.std(),
                "std_recall": recs.std(),
                "std_f1": f1s.std(),
            }
        )

    #save tensors 

    
    torch.save(accs, f"{path}/{file_name}_accs.pth")
    torch.save(precs, f"{path}/{file_name}_precs.pth")
    torch.save(recs, f"{path}/{file_name}_recs.pth")
    torch.save(f1s, f"{path}/{file_name}_f1s.pth")
    
        
        
    
    
    
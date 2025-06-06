# import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from thesis.models.model_handling import get_model
from thesis.metrics import calculate_metrics, calculate_iou
from thesis.data_handling.data_handling import get_positional_dataset, get_dataloaders
from thesis.data_handling.locate import convert_tagging2onehot
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

def test_with_crf(cfg,model,test_loader):    
    all_preds = []
    all_labels = []
    test_loss = 0
    for data in test_loader:
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)


            if "with_crf" in cfg.model.name:
                neg_likelihood, emissions = model(inputs,cfg.task.training_params.use_contrastive_loss,tags=labels)
                
                reshaped_emissions = emissions.view((emissions.shape[0]*emissions.shape[1],emissions.shape[2]))
                reshaped_labels = labels.view((emissions.shape[0]*emissions.shape[1],emissions.shape[2])).squeeze(-2)
                reshaped_labels = torch.argmax(reshaped_labels,dim=-1)
                cross_entropy_loss = torch.nn.functional.cross_entropy(reshaped_emissions,reshaped_labels )
                loss = neg_likelihood + cross_entropy_loss

                test_loss += loss.item()
            
            test_outputs = model(inputs)
            # map non_zero indice to 1 of last dimension 
            test_outputs = torch.where(torch.Tensor(test_outputs) > 0, 1, 0)
            all_preds.append(test_outputs.flatten())
            all_labels.append(convert_tagging2onehot(labels).flatten())
    test_loss /= len(test_loader)
    # Calculate metrics after each epoch
    all_preds = torch.cat(all_preds).flatten()
    all_labels = torch.cat(all_labels).flatten()
    print(all_preds.shape)
    print(all_labels.shape)
    
    print("number of 1s in preds: ",all_preds.sum())
    print("number of 1s in labels: ",all_labels.sum())

    acc, prec, rec, f1 = calculate_metrics(
        preds=all_preds, labels=all_labels
    )
    iou = calculate_iou(all_preds,all_labels)
    if cfg.wandb.use_wandb:
        wandb.log(
            data={
                "ckpt_loss":test_loss,
                "ckpt_acc": acc,
                "ckpt_precision": prec,
                "ckpt_recall": rec,
                "ckpt_f1": f1,
                "ckpt_iou": iou
            }
        )
    print("F1 Score: ",f1,"Accuracy: ",acc,"Precision: ",prec,"Recall: ",rec)
    print("test Loss: ",test_loss)
    print("IoU: ",iou)
    return acc, prec, rec, f1, iou

def test_without_crf(cfg,model,test_loader):
    all_preds = []
    all_labels = []
    val_loss = 0
    classification_loss = nn.CrossEntropyLoss().to(device)
    for i, data in enumerate(test_loader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.shape[0]
            seq_length = inputs.shape[1]
            num_llm_layers = inputs.shape[2]
            embedding_size = inputs.shape[3]
            inputs = inputs.view(batch_size * seq_length, num_llm_layers, embedding_size)
            labels = labels.view(batch_size * seq_length, -1).squeeze()
            val_outputs,encoded_space = model(inputs,cfg.task.training_params.use_contrastive_loss)
            val_loss = classification_loss(val_outputs, torch.argmax(labels,dim=-1))
            val_loss += val_loss.item()
            all_preds.append(convert_tagging2onehot(val_outputs))
            all_labels.append(convert_tagging2onehot(labels))
    val_loss /= len(test_loader)
    # Calculate metrics after each epoch
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc, prec, rec, f1 = calculate_metrics(
        preds=all_preds, labels=all_labels
    )
    iou = calculate_iou(all_preds,all_labels)
    if cfg.wandb.use_wandb:
        wandb.log(
            data={
                "ckpt_loss":val_loss,
                "ckpt_acc": acc,
                "ckpt_precision": prec,
                "ckpt_recall": rec,
                "ckpt_f1": f1
            }
        )
    print("F1 Score: ",f1,"Accuracy: ",acc,"Precision: ",prec,"Recall: ",rec)
    print("Validation Loss: ",val_loss)

    return acc, prec, rec, f1, iou


def train(cfg,model, train_loader, val_loader):

    # Loss and optimizer
    if cfg.task.training_params.use_cross_entropy_weighting:    
        #num_positive = 0
        #num_negative = 0
        #for _, labels in train_loader:
        #    num_classes = labels.shape[-1]
        #    labels = torch.argmax(labels,dim=-1).squeeze(-1)
        #    num_positive += (labels >= 1).sum().item()
        #    num_negative += (labels == 0).sum().item()
        print("hardcoded num_negatives/num_positive: ")
        num_negative = 281315
        num_positive = 9619
        num_classes = cfg.model.num_classes
        pos_weight = torch.ones(num_classes) * ((num_negative +num_positive) / num_positive)
        print("Number of positive samples: ",num_positive)
        print("Number of negative samples: ",num_negative)
        pos_weight[0] = 1.0 # set the first class to 1.0
        pos_weight = pos_weight.to(device)
    else:
        num_classes = cfg.model.num_classes
        pos_weight = torch.ones(num_classes)    
        
    print("Using Train Weight: ",pos_weight)
    classification_loss = nn.CrossEntropyLoss(weight=pos_weight).to(device)
    print("Use Contrastive Loss: ",cfg.task.training_params.use_contrastive_loss)
    contrastive_loss = ContrastiveLoss().to(device)
    print("Use Adam Optimizer with LR: ",cfg.task.training_params.learning_rate, "Weight Decay: ",cfg.task.training_params.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=cfg.task.training_params.learning_rate,weight_decay=cfg.task.training_params.weight_decay)
    print("Train on epochs: ",cfg.task.training_params.epochs)
    epochs = cfg.task.training_params.epochs 
    
    contrast_loss = torch.Tensor([0]).to(device)

    min_val_loss = torch.inf
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = (
                data  # Assuming your dataset returns input features and labels# Move inputs and labels to the device (CPU or GPU)
            )
            
            
            inputs, labels = inputs.to(device), labels.to(device)

            batch_size = inputs.shape[0]
            seq_length = inputs.shape[1]
            num_llm_layers = inputs.shape[2]
            embedding_size = inputs.shape[3]

            if cfg.task.use_downsampling:           # Downsampling logic: randomly drop indices with label 0
                labels_flat = torch.argmax(labels, dim=-1).view(-1)  # Flatten labels
                non_zero_indices = (labels_flat != 0).nonzero(as_tuple=True)[0]
                zero_indices = (labels_flat == 0).nonzero(as_tuple=True)[0]

                # Randomly sample a subset of zero indices to keep
                num_to_keep = int(len(non_zero_indices) * cfg.task.training_params.downsampling_ratio)
                if len(zero_indices) > num_to_keep:
                    zero_indices_to_keep = zero_indices[torch.randperm(len(zero_indices))[:num_to_keep]]
                    keep_indices = torch.cat([non_zero_indices, zero_indices_to_keep])
                else:
                    keep_indices = torch.arange(len(labels_flat))
                
                inputs = inputs.view(batch_size * seq_length, num_llm_layers, embedding_size)[keep_indices]
                labels = labels.view(batch_size * seq_length, -1).squeeze()[keep_indices]
                
            else:
                
                inputs = inputs.view(batch_size * seq_length, num_llm_layers, embedding_size)
                labels = labels.view(batch_size * seq_length, -1).squeeze()

            labels = torch.argmax(labels,dim=-1)

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
        

        # validation
        val_loss = 0
        
        for data in val_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                batch_size = inputs.shape[0]
                seq_length = inputs.shape[1]
                num_llm_layers = inputs.shape[2]
                embedding_size = inputs.shape[3]
                inputs = inputs.view(batch_size * seq_length, num_llm_layers, embedding_size)
                labels = labels.view(batch_size * seq_length, -1).squeeze()
                labels = torch.argmax(labels,dim=-1) 

                val_outputs,encoded_space = model(inputs,cfg.task.training_params.use_contrastive_loss)
                if cfg.task.training_params.use_contrastive_loss:
                    batch_val_loss = classification_loss(val_outputs, labels) + contrastive_loss(encoded_space,labels)
                else:
                    batch_val_loss = classification_loss(val_outputs, labels)
                val_loss += batch_val_loss.item()
        val_loss /= len(val_loader)
        # Calculate metrics after each epoch
        
        running_loss /= len(train_loader)

        if cfg.wandb.use_wandb:
            wandb.log({"train_loss":running_loss,"val_loss":val_loss})
            
  
            
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
    
    # get final evaluation
    if cfg.task.training_params.early_stopping:
        model.load_state_dict(early_stopping_checkpoint)
        test_without_crf(cfg,model,val_loader)
       
    # Save the model checkpoint
    if cfg.task.training_params.save_model:
        date = datetime.datetime.now().strftime("%H_%M__%d_%m_%Y")
        model_path = f"thesis/data/models/{cfg.model.name}_{cfg.benchmark.name}_{date}.pth"
        print("saving model to ",model_path)
        if cfg.task.training_params.early_stopping:
            torch.save(early_stopping_checkpoint,model_path)
    return model


def train_with_crf(cfg,model, train_loader, val_loader):
    
    
    #optimizer# Loss and optimizer
    if cfg.task.training_params.use_cross_entropy_weighting:    
        
        #num_positive = 0
        #num_negative = 0
        #for _, labels in train_loader:
        #    num_classes = labels.shape[-1]
        #    labels = torch.argmax(labels,dim=-1).squeeze(-1)
        #    num_positive += (labels >= 1).sum().item()
        #    num_negative += (labels == 0).sum().item()
        print("hardcoded num_negatives/num_positive: ")
        num_negative = 281315
        num_positive = 9619
        num_classes = cfg.model.num_classes
            
        pos_weight = torch.ones(num_classes) * (num_negative / num_positive)
        print("Number of positive samples: ",num_positive)
        print("Number of negative samples: ",num_negative)
        pos_weight[0] = 1.0 # set the first class to 1.0
        pos_weight = pos_weight.to(device)
    else:
        num_classes = cfg.model.num_classes
        pos_weight = torch.ones(num_classes)     
    
    print("Using Train Weight: ",pos_weight)

    print("Use Adam Optimizer with LR: ",cfg.task.training_params.learning_rate, "Weight Decay: ",cfg.task.training_params.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=cfg.task.training_params.learning_rate,weight_decay=cfg.task.training_params.weight_decay)
    print("Train on epochs: ",cfg.task.training_params.epochs)
    epochs = cfg.task.training_params.epochs 

    min_val_loss = torch.inf
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = (
                data  # Assuming your dataset returns input features and labels# Move inputs and labels to the device (CPU or GPU)
            )
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            neg_likelihood, emissions = model(inputs,cfg.task.training_params.use_contrastive_loss,tags=labels)
            if cfg.task.training_params.crf_loss_use_cross_entropy:
                reshaped_emissions = emissions.view((emissions.shape[0]*emissions.shape[1],emissions.shape[2]))
                reshaped_labels = labels.view((emissions.shape[0]*emissions.shape[1],emissions.shape[2])).squeeze(-2)
                reshaped_labels = torch.argmax(reshaped_labels,dim=-1)
                cross_entropy_loss = torch.nn.functional.cross_entropy(reshaped_emissions,reshaped_labels )
                loss = neg_likelihood + cross_entropy_loss
            else: 
                loss = neg_likelihood

            # Backward pass and optimization
            loss.backward()

            

            #print gradients
            #for name, param in model.named_parameters():
            #    print(f"{name} grad: {param.grad}")

            optimizer.step()
            # Log the loss
            running_loss += loss.item()
        

        # validation
        val_loss = 0
        
        for data in val_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                neg_likelihood, emissions = model(inputs,cfg.task.training_params.use_contrastive_loss,tags=labels)
                if cfg.task.training_params.crf_loss_use_cross_entropy:
                    reshaped_emissions = emissions.view((emissions.shape[0]*emissions.shape[1],emissions.shape[2]))
                    reshaped_labels = labels.view((emissions.shape[0]*emissions.shape[1],emissions.shape[2])).squeeze(-2)
                    reshaped_labels = torch.argmax(reshaped_labels,dim=-1)
                    cross_entropy_loss = torch.nn.functional.cross_entropy(reshaped_emissions,reshaped_labels )
                    loss = neg_likelihood + cross_entropy_loss
                else: 
                    loss = neg_likelihood

                val_loss += loss.item()

        val_loss /= len(val_loader)
        # Calculate metrics after each epoch
        
        running_loss /= len(train_loader)

        if cfg.wandb.use_wandb:
            wandb.log({"train_loss":running_loss,"val_loss":val_loss})
            
  
            
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
    
    # get final evaluation
    if cfg.task.training_params.early_stopping:
        model.load_state_dict(early_stopping_checkpoint)
        
    # Save the model checkpoint
    if cfg.task.training_params.save_model:
        date = datetime.datetime.now().strftime("%H_%M__%d_%m_%Y")
        model_path = f"thesis/data/models/{cfg.model.name}_{cfg.benchmark.name}_{date}.pth"
        print("saving model to ",model_path)
        if cfg.task.training_params.early_stopping:
            torch.save(early_stopping_checkpoint,model_path)
            
    return model 

def pipelined_training(cfg):
    init_wandb(cfg)
    # Load the dataset
    dataset = get_positional_dataset(cfg)
    # dataset = Subset(dataset,range(10))
    train_loader, val_loader, test_loader = get_dataloaders(cfg,dataset)

    print("shape of samples",dataset[0][0].shape)

    embedding_size = dataset[0][0].shape[-1]  # first batch, first input #embedding size
    num_layers = dataset[0][0].shape[-2]  # first batch, first input #embedding size
    seq_length = dataset[0][0].shape[-3]  # first batch, first input #embedding size

    print("Embedding Size", embedding_size, "Number of Layers", num_layers)

    num_output_classes = 0

    if cfg.task.tag_scheme == "IO": 
        num_output_classes = 2
    elif cfg.task.tag_scheme == "BIO":
        num_output_classes = 3
    elif cfg.task.tag_scheme == "BIOES":
        num_output_classes = 5 
    else:
        raise ValueError(f"Unknown tagging scheme: {cfg.task.tag_scheme}")
    cfg.model.num_classes = num_output_classes
    
    model = get_model(cfg,embedding_size=embedding_size,num_layers=num_layers).to(
        device)

    
    if cfg.task.use_pretrained: 
        model.load_classifier_weights(cfg.model.pretrained_model_path)

    print_number_of_parameters(model)

    from thesis.models.layer_comparison_classifier import LayerComparisonClassifier
    lcc_model = LayerComparisonClassifier(embedding_size=embedding_size, 
                                          num_llm_layers=num_layers, 
                                          output_size=cfg.model.num_classes, 
                                          layer_depth=cfg.model.layer_depth,
                                          comparison_method=cfg.model.comparison_method, 
                                          aggregation_method=cfg.model.aggregation_method,
                                          final_classifier_non_linear=cfg.model.final_classifier_non_linear)
    lcc_model = lcc_model.to(device)
    
    if cfg.task.use_pretrained: 
        lcc_model.load_classifier_weights(cfg.model.pretrained_model_path)


    cfg.task.training_params.batch_size = 100
    lcc_model = train(cfg,
        model=lcc_model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    model.load_state_dict(lcc_model.state_dict(),strict=False)
    # freeze lcc part 
    
    cfg.task.training_params.batch_size = 10
    if cfg.task.freeze_lcc: 
        for name, param in model.named_parameters():
            if "crf" in name:
                param.requires_grad = True
                continue
            param.requires_grad = False

    model = train_with_crf(cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader
        )
    
    #validate model 
    #reinit val_loader 
    cfg.task.training_params.batch_size = 1
    dataset = get_positional_dataset(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg,dataset)
    
    test_with_crf(cfg,model,val_loader)


def train_positional_layer_fusion(cfg : DictConfig):
    if cfg.task.use_pipeline:
        return pipelined_training(cfg)
    init_wandb(cfg)
    # Load the dataset
    dataset = get_positional_dataset(cfg)
    # dataset = Subset(dataset,range(10))
    train_loader, val_loader, test_loader = get_dataloaders(cfg,dataset)

    print("shape of samples",dataset[0][0].shape)

    embedding_size = dataset[0][0].shape[-1]  # first batch, first input #embedding size
    num_layers = dataset[0][0].shape[-2]  # first batch, first input #embedding size
    seq_length = dataset[0][0].shape[-3]  # first batch, first input #embedding size

    print("Embedding Size", embedding_size, "Number of Layers", num_layers)

    num_output_classes = 0

    if cfg.task.tag_scheme == "IO": 
        num_output_classes = 2
    elif cfg.task.tag_scheme == "BIO":
        num_output_classes = 3
    elif cfg.task.tag_scheme == "BIOES":
        num_output_classes = 5 
    else:
        raise ValueError(f"Unknown tagging scheme: {cfg.task.tag_scheme}")
    cfg.model.num_classes = num_output_classes
    
    model = get_model(cfg,embedding_size=embedding_size,num_layers=num_layers).to(
        device)

    
    if cfg.task.use_pretrained: 
        model.load_classifier_weights(cfg.model.pretrained_model_path)

    print_number_of_parameters(model)

    if "with_crf" in cfg.model.name:
        model = train_with_crf(cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
    else:
        model = train(cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
    try:    
        model.plot_classifier_weights()
    except:
        print("Could not plot classifier weights")
    
    #validate model 
    #reinit val_loader 
    dataset = get_positional_dataset(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg,dataset)
    cfg.task.training_params.batch_size = 1

    if "with_crf" in cfg.model.name:
        val_acc, val_prec, val_rec, val_f1, val_iou = test_with_crf(cfg,model,val_loader)
        test_acc, test_prec, test_rec, test_f1, test_iou = test_with_crf(cfg,model,test_loader)
    else: 
        val_acc, val_prec, val_rec, val_f1, val_iou = test_without_crf(cfg,model,val_loader)
        test_acc, test_prec, test_rec, test_f1, test_iou = test_without_crf(cfg,model,test_loader)
    
    val_dict = {
        "acc": val_acc,
        "prec": val_prec,
        "rec": val_rec,
        "f1": val_f1,
        "iou": val_iou
    }
    test_dict = {
        "acc": test_acc,
        "prec": test_prec,
        "rec": test_rec,
        "f1": test_f1,
        "iou": test_iou
    }

    return val_dict, test_dict


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
        all_preds > 0.0, 1.0, 0.0
    )  # Convert logits to binary predictions
    acc, prec, rec, f1 = calculate_metrics(
        preds=all_preds, labels=all_labels
    )
    
    return acc, prec, rec, f1

def average_positional_runs(cfg : DictConfig):
    
    init_wandb(cfg)
    
    saving_path = "./thesis/data/positions/"
    model_name = cfg.model.name

    # load important parameters out of config 
    tag_scheme = cfg.task.tag_scheme
    if model_name == "lcc_with_crf" or model_name == "layer_comparison_classifier":
        comparison_method = cfg.model.comparison_method
    else:
        comparison_method = "None"
    num_runs = cfg.task.number_of_runs
    if model_name == "baseline_with_crf":
        model_name = model_name + "_" + cfg.model.classifier_name
    file_name = f"{model_name}__{tag_scheme}__{comparison_method}__{cfg.task.training_params.patience}__{num_runs}.pth"
        
    if os.path.exists(f"{saving_path}/{file_name}"): 
        print("File already exists, skipping")
        return
    results = {
        "test_results":[],
        "val_results":[]
    }
    for i in tqdm(range(cfg.task.number_of_runs)):
        val_dict, test_dict = train_positional_layer_fusion(cfg)
        results["val_results"].append(val_dict)
        results["test_results"].append(test_dict)

    print("Results: ",results)    
    #save results dict 
    torch.save(results,saving_path + file_name)
    print("saved results to ",saving_path + file_name)
    
        
        
    
    
    
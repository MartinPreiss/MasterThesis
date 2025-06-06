# import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from thesis.models.model_handling import get_model
from thesis.metrics import calculate_metrics
from thesis.data_handling.data_handling import get_embedding_dataset, get_dataloaders
from thesis.utils import print_number_of_parameters, get_device, init_wandb
from thesis.data_handling.data_handling import get_refact_test_indices 

from thesis.models.loss.contrastive_loss import ContrastiveLoss
from omegaconf import DictConfig
import warnings
import wandb
import datetime

from tqdm import tqdm
import os

warnings.filterwarnings("always")

device = get_device()


def test_fake_fact_setting(cfg, model, test_loader):

    test_indices, df_ids = get_refact_test_indices(cfg)

    original_labels = [0] * len(test_indices) 
    original_labels.extend([1] * len(test_indices))
    test_indices.extend([test_indice+len(test_indices) for test_indice in test_indices])

    all_preds = []
    all_labels = []
    for i, data in enumerate(test_loader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs,encoded_space = model(inputs,cfg.task.training_params.use_contrastive_loss)
            outputs = torch.where(
                outputs >= 0.0, 1.0, 0.0)
            all_preds.extend(outputs)
            all_labels.extend(labels)
    #calculate meetrics of classifier 
    
    print("Classifier results on test set (hallucination setting)")
    acc, prec, rec, f1 = get_validation_metrics(cfg,model,test_loader)
    print(f"Test on {cfg.benchmark.name} - Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}")

    final_preds = []
    llm_outputs = []
    for i in range(len(test_indices)): 
        pred = bool(all_preds[i])
        model_did_mistake = bool(all_labels[i])
        fake_fact_in_input = bool(original_labels[i])
        
        if fake_fact_in_input: 
            llm_output = True if not model_did_mistake else False
        else: 
            llm_output = True if model_did_mistake else False
        llm_outputs.append(llm_output)

        if pred: 
            final_preds.append(not llm_output)
        else:
            final_preds.append(llm_output)
    
    # convert to tensor
    final_preds = torch.Tensor(final_preds).to(device)
    llm_outputs = torch.Tensor(llm_outputs).to(device)
    original_labels = torch.Tensor(original_labels).to(device)


    # calculate metrics  without adding mdel to prevent mistakes
    acc, prec, rec, f1 = calculate_metrics(
        preds=llm_outputs, labels=original_labels
    )

    print("Fake Fact setting result without adding model to prevent mistakes")
    print(f"Test on {cfg.benchmark.name} - Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}")

    # calculate metric with orignal labels with final preds
    acc, prec, rec, f1 = calculate_metrics(
        preds=final_preds, labels=original_labels
    )
    print("Fake Fact setting result by adding model to prevent mistakes")
    print(f"Test on {cfg.benchmark.name} - Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}")



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
    return model

def train_layer_fusion(cfg : DictConfig):
    
    init_wandb(cfg)
    # Load the dataset
    dataset = get_embedding_dataset(cfg)
    print("trainin on benchmark: ",cfg.benchmark.name)
    
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
        if cfg.model.freeze_last_layers:
            print("use pretrained model from: ",cfg.model.pretrained_model_path)
            print("freezing last layers")
            model.freeze_last_layers()
        else:
            print("no freezing of last layers")

    print_number_of_parameters(model)

    model = train(cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    try:    
        sum_fig = model.plot_classifier_weights() #sum_fig of matplotlib
        #save to diskspace 
        llm_name = cfg.llm.name[cfg.llm.name.find("/")+1:]
        file_name = f"{cfg.model.name}_{cfg.benchmark.name}_{llm_name}_{cfg.model.comparison_method}_{cfg.task.training_params.patience}"
        sum_fig.savefig(f"./thesis/data/plots/{file_name}_classifier_weights.png")
    except Exception as e:
        print(e)
        print("Could not plot classifier weights")

    #if cfg.benchmark.name == "refact":
    #    test_fake_fact_setting(cfg,model,test_loader)

    
    return model 
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

def pretrain_on_haluleval(cfg: DictConfig):

    
    cfg.task.training_params.batch_size = 100
    cfg.benchmark.name = "haluleval"
    cfg.task.use_pretrained = False
    cfg.task.training_params.learning_rate = 1e-3

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
    #save model 
    date = datetime.datetime.now().strftime("%H_%M__%d_%m_%Y")
    model_path = f"thesis/data/models/{cfg.model.name}_{cfg.benchmark.name}_{date}.pth"
    print("saving model to ",model_path)
    torch.save(model.state_dict(),model_path)

    return model_path

def average_earlystopping(cfg : DictConfig):
    
    init_wandb(cfg)

    
    
    path= cfg.task.path_to_save
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
        
        llm_name = cfg.llm.name[cfg.llm.name.find("/")+1:]

        file_name = f"{model_name}_{benchmark_name}_{llm_name}_{num_classes}_{comparison_method}_{aggregation_method}_{final_classifier_non_linear}_{layer_depth}_{contrastive_loss}_{cfg.task.training_params.patience}"
    
    else:
        layer_depth = cfg.model.layer_depth
        file_name = f"{model_name}_{benchmark_name}_{layer_depth}_{cfg.task.training_params.patience}"
    
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

    if cfg.task.use_pretrained and ((cfg.model.pretrained_model_path is None) or (cfg.model.pretrained_model_path == "None")):
        print("pretraining on haluleval benchmark")
        cfg.model.pretrained_model_path = pretrain_on_haluleval(cfg)
        cfg.task.use_pretrained = True
        
    
    cfg.wandb.use_wandb = False
    val_accs, val_precs, val_recs, val_f1s = [],[],[],[]
    accs,precs, recs, f1s = [],[],[],[]
    for i in tqdm(range(cfg.task.number_of_runs)):
        model = get_model(cfg,embedding_size=embedding_size,num_layers=num_layers).to(
        device
        )
        
        if cfg.task.use_pretrained: 
            model.load_state_dict(torch.load(cfg.model.pretrained_model_path))
            model.freeze_last_layers()
            print("use pretrained model from: ",cfg.model.pretrained_model_path)

        train(cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        #val metrics 
        val_acc, val_prec, val_rec, val_f1 = get_validation_metrics(cfg,model,val_loader)
        val_accs.append(val_acc)
        val_precs.append(val_prec)
        val_recs.append(val_rec)
        val_f1s.append(val_f1)        

        #test metrics
        acc, prec, rec, f1 = get_validation_metrics(cfg,model,test_loader)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    
    # calculate average and log for wandb 
    val_accs = torch.Tensor(val_accs)
    val_precs = torch.Tensor(val_precs)
    val_recs = torch.Tensor(val_recs)
    val_f1s = torch.Tensor(val_f1s)

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
    torch.save(val_accs, f"{path}/{file_name}_val_accs.pth")
    torch.save(val_precs, f"{path}/{file_name}_val_precs.pth")
    torch.save(val_recs, f"{path}/{file_name}_val_recs.pth")
    torch.save(val_f1s, f"{path}/{file_name}_val_f1s.pth")
    
    torch.save(accs, f"{path}/{file_name}_accs.pth")
    torch.save(precs, f"{path}/{file_name}_precs.pth")
    torch.save(recs, f"{path}/{file_name}_recs.pth")
    torch.save(f1s, f"{path}/{file_name}_f1s.pth")

def average_in_domain_shift(cfg: DictConfig):    

    
    if cfg.task.use_pretrained:
        saving_path = "thesis/data/avg_in_domain_shift_pretrained/"
        if cfg.model.freeze_last_layers:
            saving_path = "thesis/data/avg_in_domain_pretrained_freezed/"
    else:
        saving_path = "thesis/data/avg_in_domain_shift/"
    if cfg.model.name == "layer_comparison_classifier":
        comparison_method = cfg.model.comparison_method
    else: 
        comparison_method = "none"
    filename = f"{cfg.model.name}__{cfg.task.first_benchmark}__{cfg.task.second_benchmark}__{comparison_method}_results.pth"
    if os.path.exists(saving_path + filename):
        print("File already exists, skipping")
        return

    results = {
        "test_results":[],
        "val_results":[]
    }
    for i in range(cfg.task.number_of_runs):
        first_benchmark = cfg.task.first_benchmark
        second_benchmark = cfg.task.second_benchmark

        if cfg.task.use_pretrained:
            #pretrain on first benchmark
            cfg.benchmark.name = first_benchmark
            cfg.task.use_pretrained = False
            model = train_layer_fusion(cfg)

            model_path = f"thesis/data/models/{cfg.model.name}_{cfg.benchmark.name}.pth"
            print("saving model to ",model_path)
            torch.save(model.state_dict(),model_path)
            cfg.model.pretrained_model_path = model_path
            cfg.task.use_pretrained = True 

        #pretrain on second benchmark
        cfg.benchmark.name = second_benchmark

        model = train_layer_fusion(cfg)

        #get validation metrics
        dataset = get_embedding_dataset(cfg)
        train_loader, val_loader, test_loader = get_dataloaders(cfg,dataset)
        val_acc, val_prec, val_rec, val_f1 = get_validation_metrics(cfg,model,val_loader)
        results["val_results"].append({
            "acc": val_acc,
            "prec": val_prec,
            "rec": val_rec,
            "f1": val_f1
        })
        #test on all benchmarks
        test_result = test_on_all_benchmarks(cfg,model)

        results["test_results"].append(test_result)

    #save results dict 
    torch.save(results,saving_path + filename)
    print("saved results to ",saving_path + filename)

            
def test_on_all_benchmarks(cfg, model): 

    benchmarks = ["haluleval","refact","truthfulqa"]

    results = {}
    for benchmark in benchmarks:
        cfg.benchmark.name = benchmark
        # Load the dataset
        dataset = get_embedding_dataset(cfg)

        # dataset = Subset(dataset,range(10))
        train_loader, val_loader, test_loader = get_dataloaders(cfg,dataset)

        #test the model on the test set
        acc, prec, rec, f1 = get_validation_metrics(cfg,model,test_loader)
        print(f"Test on {benchmark} - Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}")

        results[benchmark] = {  
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1
        }
    return results
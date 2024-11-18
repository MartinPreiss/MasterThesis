# import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from thesis.models.lstm import LSTMModel
from thesis.metrics import calculate_metrics
from thesis.data_handling.data_handling import get_embedding_dataset, get_dataloaders
from thesis.utils import print_number_of_parameters,get_device, init_wandb
import wandb
import warnings

from tqdm import tqdm

warnings.filterwarnings("always")
device = get_device()


def train_classifier(cfg, model, train_loader, val_loader,num_layers):

    start = 0 #num_layers // 2
    end = num_layers - 0 
    
    num_layers = len(range(start,end))
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training_params.learning_rate)
    
    # training loop
    max_f1 = 0
    for epoch in tqdm(range(cfg.training_params.epochs)):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data 
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs[:,start:end,:]
            labels = labels#.unsqueeze(1).repeat(1,num_layers,1)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        running_loss /= len(train_loader)
        # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")
        if cfg.wandb.use_wandb:
            wandb.log({"train_loss":running_loss})

        # validation
        all_preds = []
        all_labels = []
        val_loss = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                inputs = inputs[:,start:end,:]
                labels = labels#.unsqueeze(1).repeat(1,num_layers,1)
                val_outputs = model(inputs)
                loss = criterion(val_outputs, labels)
                val_loss += loss.item()
                all_preds.append(val_outputs)
                all_labels.append(labels)

        val_loss /= len(val_loader)
        # Calculate metrics after each epoch
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_preds = torch.where(all_preds >= 0.0, 1.0, 0.0)  # Convert logits to binary predictions

        #for layer_id in range(num_layers):
        acc, prec, rec, f1 = calculate_metrics(preds=all_preds, labels=all_labels)
        if cfg.wandb.use_wandb:
            wandb.log(data={"val_acc": acc, "val_loss": val_loss, "val_precision": prec, "val_recall": rec, "f1": f1})
        # Save the model checkpoint
        # torch.save(model.state_dict(), "simple_classifier.pth")
        max_f1 = f1 if f1>max_f1 else max_f1
    if cfg.wandb.use_wandb:
        wandb.log({"max_f1":max_f1})
    
def prepare_and_start_training(cfg : DictConfig):
    
    
    # start a new wandb run to track this script
    init_wandb(cfg)

    # Load the dataset
    dataset = get_embedding_dataset(cfg)
    
    # dataset = Subset(dataset,range(10))
    train_loader, val_loader, test_loader = get_dataloaders(cfg,dataset)

    input_size = dataset[0][0].shape[-1]  # first batch, first input #embedding size
    num_layers = dataset[0][0].shape[-2]  # first batch, first input #embedding size
    print("Embedding Size", input_size, "Number of Layers", num_layers)

    model = LSTMModel(input_size,hidden_size=cfg.lstm.hidden_size,num_layers=cfg.lstm.num_layers).to(device)

    print_number_of_parameters(model)

    train_classifier(cfg=cfg,model=model,train_loader=train_loader,val_loader=val_loader,num_layers = num_layers)

if __name__ == "__main__":
    prepare_and_start_training()
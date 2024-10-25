# import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

from thesis.models.lstm import LSTMModel
from thesis.metrics import calculate_metrics
from thesis.data_handling import get_embedding_dataset, get_dataloaders, PCADataset
from thesis.utils import print_number_of_parameters,get_device, get_config_and_init_wandb

import wandb
import warnings

from tqdm import tqdm

warnings.filterwarnings("always")
# start a new wandb run to track this script
cfg = get_config_and_init_wandb(name="LSTM")
device = get_device()


def train_classifier(model, train_loader, val_loader,num_layers):

    start = num_layers // 2
    end = num_layers - 0 
    
    num_layers = len(range(start,end))
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training_params.learning_rate)
    
    # training loop
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
        wandb.log(data={"val_acc": acc, "val_loss": val_loss, "val_precision": prec, "val_recall": rec, "f1": f1})
        # Save the model checkpoint
        # torch.save(model.state_dict(), "simple_classifier.pth")


if __name__ == "__main__":

    # Load the dataset
    dataset = get_embedding_dataset(cfg)
    
    # dataset = Subset(dataset,range(10))
    train_loader, val_loader = get_dataloaders(dataset)

    input_size = dataset[0][0].shape[-1]  # first batch, first input #embedding size
    num_layers = dataset[0][0].shape[-2]  # first batch, first input #embedding size
    print("Embedding Size", input_size, "Number of Layers", num_layers)

    model = LSTMModel(input_size,hidden_size=cfg.lstm.hidden_size,num_layers=cfg.lstm.num_layers).to(device)

    print_number_of_parameters(model)

    train_classifier(model=model,train_loader=train_loader,val_loader=val_loader,num_layers = num_layers)
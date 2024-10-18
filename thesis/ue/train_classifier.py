#import wandb 
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import wandb
from torch.utils.data import random_split

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="thesis",
    name= None,

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "epochs": 50,
    "model_name": "gemma-2-9b-it",
    "layer_ids": [20]
    }
)


def calculate_metrics(predictions, labels):
    preds = torch.round(torch.sigmoid(predictions)).detach().cpu().numpy()  # Convert logits to binary predictions
    labels = labels.detach().cpu().numpy()  # Move labels to CPU and convert to numpy

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    return accuracy, precision, recall, f1



class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__": 
    
    # Load the dataset
    dataset = torch.load(f"./data/datasets/embeddings/embedding_{wandb.config.model_name}_{wandb.config.layer_ids}.pth")   
    
    dataset = Subset(dataset,range(10))

    # Split the dataset
    data_size = len(dataset)
    train_size = int(0.8 * data_size)
    val_size = data_size - train_size 
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=False)

    
    #loading classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = dataset[0][0].shape[-1] #first batch, first input 
    model = SimpleClassifier(input_size).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(),lr=wandb.config.learning_rate)
    epochs = wandb.config.epochs

    #training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data  # Assuming your dataset returns input features and labels# Move inputs and labels to the device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Log the loss
            running_loss += loss.item()
            if i % 10 == 9:    # log every 10 mini-batches
                wandb.log({ "train_loss_batch": loss})
        
        #validation 
        all_preds = []
        all_labels = []
        val_loss = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_preds.append(outputs)
                all_labels.append(labels)
        val_loss /= len(val_loader)        
        # Calculate metrics after each epoch
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc, prec, rec, f1 = calculate_metrics(all_preds, all_labels)
        
        running_loss /= len(train_loader)
        wandb.log({"train_loss":running_loss,"val_acc": acc, "val_loss": val_loss, "val_precision": prec, "val_recall": rec, "f1": f1})
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")

    # Save the model checkpoint
    torch.save(model.state_dict(), "simple_classifier.pth")
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
    "layer_ids": [20],
    "num_fnn_layers": 1
    }
)


def calculate_metrics(predictions, labels):
    preds =torch.where(predictions>=0.0,1.0,0.0).detach().cpu().numpy()  # Convert logits to binary predictions
    labels = labels.detach().cpu().numpy()  # Move labels to CPU and convert to numpy

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)

    return accuracy, precision, recall, f1



class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_layers=3):
        super(SimpleClassifier, self).__init__()
        
        # List to hold the layers
        self.layers = nn.ModuleList()
        current_size = input_size
        
        # Dynamically create hidden layers
        for i in range(num_layers - 1):
            next_size = current_size // 2  # Halving the size each time
            self.layers.append(nn.Linear(in_features=current_size, out_features=next_size))
            current_size = next_size
        
        # Final output layer (reducing to 1 or a predefined output size)
        self.output_layer = nn.Linear(in_features=current_size, out_features=1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        
        x = self.output_layer(x)
        return x

if __name__ == "__main__": 
    
    # Load the dataset
    dataset = torch.load(f"./data/datasets/embeddings/embedding_{wandb.config.model_name}_{wandb.config.layer_ids}.pth")   
    
    #dataset = Subset(dataset,range(10))

    # Split the dataset
    data_size = len(dataset)
    train_size = int(0.8 * data_size)
    val_size = data_size - train_size 
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("Dataset Size", data_size)
    print("Trainset Size", train_size)
    print("Valset Size", val_size)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=False)

    
    #loading classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = dataset[0][0].shape[-1] #first batch, first input
    print("Embedding Size", input_size) 
    model = SimpleClassifier(input_size,num_layers=wandb.config.num_fnn_layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
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
                val_outputs = model(inputs)
                loss = criterion(val_outputs, labels)
                val_loss += loss.item()
                all_preds.append(val_outputs)
                all_labels.append(labels)
        val_loss /= len(val_loader)        
        # Calculate metrics after each epoch
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc, prec, rec, f1 = calculate_metrics(predictions=all_preds, labels=all_labels)
        
        running_loss /= len(train_loader)
        wandb.log({"train_loss":running_loss,"val_acc": acc, "val_loss": val_loss, "val_precision": prec, "val_recall": rec, "f1": f1})
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")

    # Save the model checkpoint
    torch.save(model.state_dict(), "simple_classifier.pth")
# import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

from thesis.models.neural_net import SimpleClassifier
from thesis.metrics import calculate_metrics
from thesis.data_handling import get_embedding_dataset, get_dataloaders
from thesis.utils import print_number_of_parameters

import wandb
import warnings

from tqdm import tqdm

warnings.filterwarnings("always")
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="thesis",
    name=None,
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "epochs": 50,
        "model_name": "gemma-2-27b-it",
        "num_fnn_layers": 3,
    },
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier(model, train_loader, val_loader, num_layers):

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    epochs = wandb.config.epochs

    for layer_id in tqdm(range(num_layers)):
        max_acc, max_prec, max_rec, max_f1 = 0, 0, 0, 0
        # training loop
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = (
                    data  # Assuming your dataset returns input features and labels# Move inputs and labels to the device (CPU or GPU)
                )
                inputs = inputs[:, layer_id, :]
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

            # validation
            all_preds = []
            all_labels = []
            val_loss = 0
            for i, data in enumerate(val_loader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs = inputs[:, layer_id, :]
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

            all_preds = torch.where(
                all_preds >= 0.0, 1.0, 0.0
            )  # Convert logits to binary predictions
            acc, prec, rec, f1 = calculate_metrics(
                predictions=all_preds, labels=all_labels
            )
            if f1 > max_f1:
                max_prec = prec
                max_acc = acc
                max_rec = rec
                max_f1 = f1

            running_loss /= len(train_loader)
            # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")
            # wandb.log({"train_loss":running_loss,"val_acc": acc, "val_loss": val_loss, "val_precision": prec, "val_recall": rec, "f1": f1})

        wandb.log(
            data={
                "val_acc": max_acc,
                "val_precision": max_prec,
                "val_recall": max_rec,
                "f1": max_f1,
            },
            step=layer_id + 1,
        )

    # Save the model checkpoint
    # torch.save(model.state_dict(), "simple_classifier.pth")


if __name__ == "__main__":

    # Load the dataset
    dataset = get_embedding_dataset(wandb.config.model_name)

    # dataset = Subset(dataset,range(10))
    train_loader, val_loader = get_dataloaders(dataset)

    input_size = dataset[0][0].shape[-1]  # first batch, first input #embedding size
    num_layers = dataset[0][0].shape[-2]  # first batch, first input #embedding size
    print("Embedding Size", input_size, "Number of Layers", num_layers)

    model = SimpleClassifier(input_size, num_layers=wandb.config.num_fnn_layers).to(
        device
    )

    print_number_of_parameters(model)

    train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_layers=num_layers,
    )

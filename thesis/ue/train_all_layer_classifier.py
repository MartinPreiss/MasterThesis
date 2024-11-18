# import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from thesis.models.neural_net import SimpleClassifier,AllLayerClassifier
from thesis.metrics import calculate_metrics
from thesis.data_handling.data_handling import get_embedding_dataset, get_dataloaders, PCADataset
from thesis.utils import print_number_of_parameters, get_device, init_wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import warnings
import wandb

from tqdm import tqdm

warnings.filterwarnings("always")

device = get_device()


def train_classifier(cfg,model, train_loader, val_loader, num_layers):

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training_params.learning_rate,weight_decay=cfg.training_params.weight_decay)
    epochs = cfg.training_params.epochs

    max_f1 = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = (
                data  # Assuming your dataset returns input features and labels# Move inputs and labels to the device (CPU or GPU)
            )
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
        if cfg.wandb.use_wandb:
            wandb.log({"train_loss":running_loss})
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
        all_preds = torch.where(
            all_preds >= 0.0, 1.0, 0.0
        )  # Convert logits to binary predictions
        acc, prec, rec, f1 = calculate_metrics(
            preds=all_preds, labels=all_labels
        )
        running_loss /= len(train_loader)
        # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")
        #if cfg.wandb.use_wandb:
            #wandb.log({"train_loss":running_loss,"val_acc": acc, "val_loss": val_loss, "val_precision": prec, "val_recall": rec, "f1": f1})
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

@hydra.main(config_path="../config", config_name="config")
def prepare_and_start_training(cfg : DictConfig):
    
    init_wandb(cfg)
    # Load the dataset
    dataset = get_embedding_dataset(cfg)
    
    # dataset = Subset(dataset,range(10))
    train_loader, val_loader, test_loader = get_dataloaders(cfg,dataset)

    input_size = dataset[0][0].shape[-1]  # first batch, first input #embedding size
    num_layers = dataset[0][0].shape[-2]  # first batch, first input #embedding size
    print("Embedding Size", input_size, "Number of Layers", num_layers)

    model = AllLayerClassifier(input_size, num_llm_layers=num_layers).to(
        device
    )

    print_number_of_parameters(model)

    train_classifier(cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_layers=num_layers,
    )


if __name__ == "__main__":
    prepare_and_start_training()

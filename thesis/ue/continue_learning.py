# import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from thesis.models.model_handling import get_model
from thesis.metrics import calculate_metrics
from thesis.data_handling.data_handling import get_embedding_dataset, get_dataloaders
from thesis.utils import print_number_of_parameters, get_device, init_wandb
from omegaconf import DictConfig
import warnings
import wandb

from tqdm import tqdm

warnings.filterwarnings("always")

device = get_device()


def train(cfg,model, train_loader, val_loader, num_layers):

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.task.training_params.learning_rate, weight_decay=cfg.task.training_params.weight_decay)
    epochs = cfg.task.training_params.epochs

    max_f1 = 0
    for layer_id in tqdm(range(num_layers)):
        acc, prec, rec, f1 = 0, 0, 0, 0
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
            tmp_acc, tmp_prec, tmp_rec, tmp_f1 = calculate_metrics(
                preds=all_preds, labels=all_labels
            )
            if tmp_f1 > f1:
                prec = tmp_prec
                acc = tmp_acc
                rec = tmp_rec
                f1 = tmp_f1

            running_loss /= len(train_loader)
            # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")
        #if cfg.use_wandb:
        #if cfg.wandb.use_wandb:
        #    wandb.log({"train_loss":running_loss,"val_acc": acc, "val_loss": val_loss, "val_precision": prec, "val_recall": rec, "f1": f1})
        max_f1 = f1 if f1 > max_f1 else max_f1
        if cfg.wandb.use_wandb:
            wandb.log(
            data={
                "val_acc": acc,
                "val_precision": prec,
                "val_recall": rec,
                "f1": f1,
            },
            step=layer_id + 1,
        )
    if cfg.wandb.use_wandb:
        if cfg.wandb.use_wandb:
            wandb.log({"max_f1":max_f1}) 
    # Save the model checkpoint
    # torch.save(model.state_dict(), "simple_classifier.pth")

def continue_learning(cfg : DictConfig):

    init_wandb(cfg)
    # Load the dataset
    dataset = get_embedding_dataset(cfg)

    # dataset = Subset(dataset,range(10))
    train_loader, val_loader, test_loader = get_dataloaders(cfg,dataset)

    embedding_size = dataset[0][0].shape[-1]  # first batch, first input #embedding size
    num_layers = dataset[0][0].shape[-2]  # first batch, first input #embedding size
    print("Embedding Size", embedding_size, "Number of Layers", num_layers)

    model = get_model(embedding_size, num_layers=cfg.task.nn.num_layers).to(
        device
    )

    print_number_of_parameters(model)

    train(cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_layers=num_layers,
    )
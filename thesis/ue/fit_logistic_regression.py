from sklearn.linear_model import LogisticRegression
import torch 
from sklearn.model_selection import train_test_split
import wandb

from thesis.utils import get_config_and_init_wandb
from thesis.data_handling import get_embedding_dataset
from thesis.metrics import calculate_metrics

cfg  = get_config_and_init_wandb(name="Log_Reg")

# Load the dataset 
dataset = get_embedding_dataset(cfg)
#dataset = Subset(dataset,range(10))
# Split the dataset
data_size = len(dataset)
print(len(dataset))

# Extract all inputs and labels
inputs_list, labels_list = [], []

for data in dataset:
    inputs, labels = data
    inputs_list.append(inputs)
    labels_list.append(labels)

# Stack the inputs and labels into a single tensor
all_inputs = torch.stack(inputs_list)
all_labels = torch.stack(labels_list)

# Convert tensors to numpy arrays
all_inputs_np = all_inputs.cpu().numpy()
all_labels_np = all_labels.cpu().numpy()

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(all_inputs_np, all_labels_np, test_size=0.2, random_state=cfg.seed)

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=200)

num_layers = X_train.shape[-2]

for layer_id in range(num_layers):
    # Train the model using the training data
    log_reg.fit(X_train[:,layer_id,:], y_train)
    
    acc, prec, rec, f1 =  calculate_metrics(log_reg.predict(X_test[:,layer_id,:]),y_test)
    wandb.log(data={"val_acc": acc, "val_precision": prec, "val_recall": rec, "f1": f1},step=layer_id+1)

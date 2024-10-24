from sklearn.linear_model import LogisticRegression
#import  
import torch 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="thesis",
    name= "Log_Reg_all_layers_gemma27",

    # track hyperparameters and run metadata
    config={
    "model_name": "gemma-2-27b-it",
    "layer_ids": "all",
    }
)
    
def calculate_metrics(preds, labels):

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)

    return accuracy, precision, recall, f1
    

# Load the dataset
dataset = torch.load(f"./data/datasets/embeddings/embedding_{wandb.config['model_name']}_{wandb.config['layer_ids']}.pth")   

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
X_train, X_test, y_train, y_test = train_test_split(all_inputs_np, all_labels_np, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=200)

num_layers = X_train.shape[-2]

for layer_id in range(num_layers):
    # Train the model using the training data
    log_reg.fit(X_train[:,layer_id,:], y_train)
    
    acc, prec, rec, f1 =  calculate_metrics(log_reg.predict(X_test[:,layer_id,:]),y_test)
    wandb.log(data={"val_acc": acc, "val_precision": prec, "val_recall": rec, "f1": f1},step=layer_id+1)

from sklearn.linear_model import LogisticRegression

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np 
from sklearn.model_selection import train_test_split
import wandb

from thesis.data_handling.data_handling import get_embedding_dataset
from thesis.metrics import calculate_metrics
from thesis.utils import init_wandb

class Log_Reg_Ensemble():
    def __init__(self, num_layers,start_layer=0):
        
        self.classifiers = [LogisticRegression() for layer_id in range(num_layers) ]
        self.start_layer= start_layer
    
    def fit(self,X_train,y_train):
        for layer_id, classifier in enumerate(self.classifiers):
            if layer_id < self.start_layer:
                continue
            classifier.fit(X_train[:,layer_id,:], y_train)
            
    
    def predict(self,X_test):
        result = 0
        for layer_id, classifier in enumerate(self.classifiers):
            if layer_id < self.start_layer:
                continue
            result += classifier.predict(X_test[:,layer_id,:])
        final_result = np.where(result>= (len(self.classifiers)-self.start_layer)/2,1,0)
        return final_result


@hydra.main(config_path="../config", config_name="config")
def prepare_and_start_training(cfg : DictConfig):
    
    init_wandb(cfg)
    dataset = get_embedding_dataset(cfg)
    
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


    num_layers = X_train.shape[-2]
    # Initialize Logistic Regression model
    log_reg = Log_Reg_Ensemble(num_layers,start_layer=21)

    log_reg.fit(X_train, y_train)
    acc, prec, rec, f1 =  calculate_metrics(log_reg.predict(X_test),y_test)
    if cfg.use_wandb:
        wandb.log(data={"val_acc": acc, "val_precision": prec, "val_recall": rec, "f1": f1})

if __name__ == "__main__":
    prepare_and_start_training()
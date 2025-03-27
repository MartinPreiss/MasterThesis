from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
import torch 

def calculate_metrics(preds, labels):
    if isinstance(preds,torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()  # Move labels to CPU and convert to numpy

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)

    return accuracy, precision, recall, f1

def calculate_score_metrics(scores,labels): 
    #calculate metrics without thresholding like AUROC or  AUPRC 
    if isinstance(scores,torch.Tensor):
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()  # Move labels to CPU and convert to numpy
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(labels, scores)
    # Calculate AUPRC
    average_precision = average_precision_score(labels, scores)
    
    return roc_auc, average_precision
    
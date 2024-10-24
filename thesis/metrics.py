from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()  # Move labels to CPU and convert to numpy

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)

    return accuracy, precision, recall, f1
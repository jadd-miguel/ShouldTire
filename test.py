import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from model import DEVICE

def test_model(train_loss, train_accuracy, val_loss, val_accuracy, model, ts_loader):

    plt.figure()
    plt.plot(train_loss, color='green', label='Training Loss')
    plt.plot(val_loss, color='red', label='Validation Loss')
    plt.title('Loss Plot')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    _ = plt.legend()

    plt.figure()
    plt.plot(train_accuracy, color='green', label='Training Accuracy')
    plt.plot(val_accuracy, color='red', label='Validation Accuracy')
    plt.title('Accuracy Plot')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    _ = plt.legend()

    y_true = []
    y_pred = []
    for i, batch in enumerate(ts_loader):

        inputs, labels = (batch[0].to(DEVICE), batch[1].to(DEVICE))
        y_true.extend(labels.detach().cpu().numpy())

        output = model(inputs)
        _, preds = torch.max(output, 1)
        y_pred.extend(preds.detach().cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"ACCURACY IS:  {accuracy}")
    print(f"PRECISION IS: {precision}")
    print(f"RECALL IS:    {recall}")
        
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,10))
    _ = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

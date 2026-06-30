import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from util import get_device

def test_model(trainHistory, model, loader):

    def print_some_graphs():
        plt.figure()
        plt.plot(train_loss, color='green', label='Training Loss')
        plt.plot(val_loss, color='red', label='Validation Loss')
        plt.title('Loss Plot')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("graphs/loss_plot.png")
        plt.close()

        plt.figure()
        plt.plot(train_accuracy, color='green', label='Training Accuracy')
        plt.plot(val_accuracy, color='red', label='Validation Accuracy')
        plt.title('Accuracy Plot')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("graphs/acc_plot.png")
        plt.close()

    def print_conf_matrix(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.savefig("graphs/conf_matrix.png")
        plt.close()

    def test():
        y_true = []
        y_pred = []
        for _, batch in enumerate(loader.get_ts_loader()):

            inputs, labels = (batch[0].to(get_device()), batch[1].to(get_device()))
            y_true.extend(labels.detach().cpu().numpy())

            output = model(inputs)
            _, preds = torch.max(output, 1)
            y_pred.extend(preds.detach().cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1_score = 2 * (precision * recall) / (precision + recall)

        final_output = f'''
ACCURACY IS:  {accuracy}
PRECISION IS: {precision}
RECALL IS:    {recall}
F1 Score IS:  {f1_score}
'''

        print_conf_matrix(y_true, y_pred)
        return final_output

    train_loss = trainHistory.get_tr_loss()
    train_accuracy = trainHistory.get_tr_loss()
    val_loss = trainHistory.get_tr_loss()
    val_accuracy = trainHistory.get_tr_loss()

    output = test()
    print_some_graphs()

    return output

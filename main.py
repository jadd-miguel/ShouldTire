import torch.nn as nn
import torch.optim as optim
from get_data import get_data
from preprocess import preprocess
from model import get_model
from train import train
from test import test_model

def main():
    tr_loader, val_loader, ts_loader = get_data()
    tr_loader, val_loader, ts_loader = preprocess(tr_loader, val_loader, ts_loader)

    model = get_model()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_loss, train_accuracy, val_loss, val_accuracy = train(
        tr_loader,
        val_loader,
        model,
        optimizer,
        criterion
    )

    test_model(train_loss, train_accuracy, val_loss, val_accuracy, model, ts_loader)

if __name__ == "__main__":
    main()

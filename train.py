import torch
from model import DEVICE

def train(tr_loader, val_loader, model, optimizer, criterion):
    EPOCHS = 4

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    for epoch in range(EPOCHS):

        for i, batch in enumerate(tr_loader):

            inputs, labels = (batch[0].to(DEVICE), batch[1].to(DEVICE))
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            _, preds = torch.max(outputs, 1)
            accuracy = torch.sum(preds == labels.data).item() / len(labels)
            train_accuracy.append(accuracy)

            if i % 250 == 0:
                print(f"(TRAIN) EPOCH-{epoch} | ITER-{i} | LOSS-{round(loss.item(), 3)} | ACC-{round(accuracy, 3)}")

        model.eval()
        for i, batch in enumerate(val_loader):

            inputs, labels = (batch[0].to(DEVICE), batch[1].to(DEVICE))

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss.append(loss.item())
            _, preds = torch.max(outputs, 1)
            accuracy = torch.sum(preds == labels.data).item() / len(labels)
            val_accuracy.append(accuracy)

            if i % 250 == 0:
                print(f"(VAL)   EPOCH-{epoch} | ITER-{i} | LOSS-{round(loss.item(), 3)} | ACC-{round(accuracy, 3)}")

    return train_loss, train_accuracy, val_loss, val_accuracy

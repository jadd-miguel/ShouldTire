import torch
from modelling.util.types import TrainingHistory
from util import get_device

def train_model(loader, model, optimizer, criterion):
    EPOCHS = 3

    output = ""
    tr_loss = []
    tr_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(EPOCHS):

        for i, batch in enumerate(loader.get_tr_loader()):

            inputs, labels = (batch[0].to(get_device()), batch[1].to(get_device()))
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            tr_loss.append(loss.item())
            _, preds = torch.max(outputs, 1)
            accuracy = torch.sum(preds == labels.data).item() / len(labels)
            tr_acc.append(accuracy)

            if i % 25 == 0:
                log = f"(TRAIN) EPOCH-{epoch} | ITER-{i} | LOSS-{round(loss.item(), 5)} | ACC-{round(accuracy, 5)}"
                output += f"{log}\n"
                print(log)

        model.eval()
        for i, batch in enumerate(loader.get_val_loader()):

            inputs, labels = (batch[0].to(get_device()), batch[1].to(get_device()))

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss.append(loss.item())
            _, preds = torch.max(outputs, 1)
            accuracy = torch.sum(preds == labels.data).item() / len(labels)
            val_acc.append(accuracy)

            if i % 10 == 0:
                log = f"(VAL)   EPOCH-{epoch} | ITER-{i} | LOSS-{round(loss.item(), 5)} | ACC-{round(accuracy, 5)}"
                output += f"{log}\n"
                print(log)

    return TrainingHistory(output, tr_loss, tr_acc, val_loss, val_acc)

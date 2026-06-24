import torch
import torch.utils.data as util
from torchvision import datasets

# The dataset is a balanced set of good and tires, around 1000 images each
# https://www.kaggle.com/datasets/warcoder/tyre-quality-classification

BATCH_SIZE = 32

def get_data():
    dataset = datasets.ImageFolder("./TIRE_DATASET")

    train, val, test = util.random_split(
        dataset,
        [1300, 370, 186],
        generator=torch.Generator().manual_seed(31)
    )

    tr_loader = util.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = util.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    ts_loader = util.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    return tr_loader, val_loader, ts_loader

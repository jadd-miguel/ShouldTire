from torchvision import transforms

TR_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

VAL_TS_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess(tr_loader, val_loader, ts_loader):
    tr_loader.dataset.dataset.transform = TR_TRANSFORM
    val_loader.dataset.dataset.transform = VAL_TS_TRANSFORM
    ts_loader.dataset.dataset.transform = VAL_TS_TRANSFORM

    return tr_loader, val_loader, ts_loader

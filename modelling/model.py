import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model():

    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in resnet50.parameters():
        param.requires_grad = False

    NUM_OF_CLASSES = 2
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, NUM_OF_CLASSES)
    resnet50 = resnet50.to(DEVICE)

    return resnet50

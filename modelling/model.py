import torch.nn as nn
from torchvision import models
from util import get_device

def get_model():

    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in resnet50.parameters():
        param.requires_grad = False

    NUM_OF_CLASSES = 2
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, NUM_OF_CLASSES)
    resnet50 = resnet50.to(get_device())

    return resnet50

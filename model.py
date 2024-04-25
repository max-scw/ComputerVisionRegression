from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torchvision.transforms as transforms


from typing import Union, List, Dict


def build_model():
    # Load a pre-trained model
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # # Remove the classification layer (we only want the feature extractor)
    # model = nn.Sequential(*list(model.children())[:-1])
    # freeze model parameter/weights of descriptor
    for param in model.parameters():
        param.requires_grad = False
    # replace existing (classifier) head with regression head
    # (parameters / weights of newly constructed modules have requires_grad=True by default)
    # num_features = mdl.fc.in_features
    num_features = model.classifier[0].in_features
    model.classifier = torch.nn.Sequential(
        # nn.BatchNorm2d(num_features),
        torch.nn.ReLU(),
        # torch.nn.Linear(in_features=num_features, out_features=1024, bias=False),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=num_features, out_features=1, bias=True)
    )

    # set up preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, preprocess

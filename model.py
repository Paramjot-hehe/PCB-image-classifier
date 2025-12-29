import torch.nn as nn
from torchvision import models

def get_pcb_model(num_classes):
    # 1. Load the pre-trained ResNet-50
    # This model was trained on 1 million images to recognize shapes/textures
    model = models.resnet50(weights='IMAGENET1K_V1')

    # 2. Update the 'Head'
    # The original ResNet looks for 1000 things. We change it to look for 6 things.
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),              # Helps prevent memorization
        nn.Linear(512, num_classes)   # Output layer for your 6 categories
    )
    
    return model
import torch.nn as nn
from torchvision import models

def get_efficientnet_b3_model(num_classes=13):
    model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

model = get_efficientnet_b3_model(num_classes=13)

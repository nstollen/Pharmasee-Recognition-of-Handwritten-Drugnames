import torch
import torch.nn as nn
import torchvision.models as models


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # Load pretrained ResNet18 as backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify first conv to accept 1-channel grayscale input
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove final classification layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # New classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, 32, 128)
        x = self.features(x)     # (batch, 512, 1, 1)
        x = x.flatten(1)         # (batch, 512)
        x = self.classifier(x)   # (batch, num_classes)
        return x
import torch
import torch.nn as nn
import torchvision.models as models

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Reference: 'Squeeze-and-Excitation Networks' (CVPR 2018)
    Role: Recalibrates channel-wise feature responses explicitly.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResEmoteNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ResEmoteNet, self).__init__()
        # 1. Load ResNet Backbone
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. Modify for Grayscale (FER-2013 is 48x48 grayscale)
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 3. Add SE Attention Blocks (The "Novelty")
        # We inject SE blocks after each major ResNet layer
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)
        
        # 4. Final Classification Layer
        self.base.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial Conv
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        # Encoder Blocks with Attention
        x = self.base.layer1(x)
        x = self.se1(x)       # <--- Attention Added
        
        x = self.base.layer2(x)
        x = self.se2(x)       # <--- Attention Added
        
        x = self.base.layer3(x)
        x = self.se3(x)       # <--- Attention Added
        
        x = self.base.layer4(x)
        x = self.se4(x)       # <--- Attention Added

        # Classification
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.fc(x)
        return x

if __name__ == "__main__":
    # Sanity Check: Run this file to see if the model builds without error
    model = ResEmoteNet()
    print(model)
    dummy_input = torch.randn(1, 1, 48, 48)
    print(f"Output shape: {model(dummy_input).shape}")
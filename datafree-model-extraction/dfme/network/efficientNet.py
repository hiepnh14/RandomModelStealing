import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Generalized EfficientNetV2 model
class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=10, architecture_config=None):
        super(EfficientNetV2, self).__init__()
        # Define the stem of EfficientNetV2 models
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True)
        )
        
        # Define MBConv blocks based on architecture_config
        self.blocks = self._make_blocks(architecture_config)
        
        # Conv head
        self.conv_head = nn.Sequential(
            nn.Conv2d(architecture_config[-1]['out_channels'], 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        
        # Adaptive pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)
    
    def _make_blocks(self, config):
        layers = []
        in_channels = 24
        for layer in config:
            layers.append(self._make_block(in_channels, layer['out_channels'], layer['expansion_factor'], layer['stride'], layer['num_blocks']))
            in_channels = layer['out_channels']
        return nn.Sequential(*layers)

    def _make_block(self, in_channels, out_channels, expansion_factor, stride, num_blocks):
        layers = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1  # Only the first block in the series will have the specified stride
            layers.append(self._mbconv_block(in_channels, out_channels, expansion_factor, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _mbconv_block(self, in_channels, out_channels, expansion_factor, stride):
        expanded_channels = in_channels * expansion_factor
        layers = []
        # 1x1 pointwise conv to expand channels
        if expansion_factor != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU6(inplace=True))
        # 3x3 depthwise conv
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU6(inplace=True))
        # 1x1 pointwise conv to project back to out_channels
        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # Shortcut connection for residuals
        if stride == 1 and in_channels == out_channels:
            return nn.Sequential(*layers, nn.Identity())
        else:
            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Configuration dictionary for each model variant (V2-S, V2-M, V2-L)
efficientnet_v2_configs = {
    'efficientnet_v2_s': [
        {'out_channels': 48, 'expansion_factor': 2, 'stride': 1, 'num_blocks': 3},
        {'out_channels': 80, 'expansion_factor': 4, 'stride': 2, 'num_blocks': 4},
        {'out_channels': 160, 'expansion_factor': 4, 'stride': 2, 'num_blocks': 4},
    ],
    'efficientnet_v2_m': [
        {'out_channels': 48, 'expansion_factor': 3, 'stride': 1, 'num_blocks': 3},
        {'out_channels': 80, 'expansion_factor': 6, 'stride': 2, 'num_blocks': 5},
        {'out_channels': 160, 'expansion_factor': 6, 'stride': 2, 'num_blocks': 5},
        {'out_channels': 176, 'expansion_factor': 6, 'stride': 2, 'num_blocks': 5},
    ],
    'efficientnet_v2_l': [
        {'out_channels': 48, 'expansion_factor': 4, 'stride': 1, 'num_blocks': 4},
        {'out_channels': 80, 'expansion_factor': 6, 'stride': 2, 'num_blocks': 7},
        {'out_channels': 160, 'expansion_factor': 6, 'stride': 2, 'num_blocks': 7},
        {'out_channels': 176, 'expansion_factor': 6, 'stride': 2, 'num_blocks': 7},
        {'out_channels': 224, 'expansion_factor': 6, 'stride': 1, 'num_blocks': 6},
    ]
}


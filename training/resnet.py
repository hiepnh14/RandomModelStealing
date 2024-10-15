import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# 
#  BasicBlock represents the basic building block of the ResNet architecture, 
# consisting of two convolutional layers with Batch Normalization and ReLU activation.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Shortcut connection to match the input and output dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Element-wise addition between the output and the shortcut
        out = F.relu(out)
        return out
# The ResNet class constructs the network using these blocks, with an 
    # initial convolutional layer followed by three layers of basic blocks.
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # Three layers of basic blocks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # Final fully connected layer
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
# 3 blocks for each layer to form a 20-layer ResNet
def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])

# Define ResNet18
def ResNet18():
    # ResNet18 has [2, 2, 2, 2] residual blocks in 4 layers
    return ResNet(BasicBlock, [2, 2, 2])

# Define ResNet34
def ResNet34():
    # ResNet34 has [3, 4, 6, 3] residual blocks in 4 layers
    return ResNet(BasicBlock, [3, 4, 6])

# A simpler variant than ResNet18 (ResNet10)
def ResNet10():
    # This has [1, 1, 1] residual blocks in 3 layers
    return ResNet(BasicBlock, [1, 1, 1])
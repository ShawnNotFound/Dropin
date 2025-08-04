import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_c, dropin=False, num_classes=1000):
        super(ResNet, self).__init__()
        self.dropin = dropin
        self.in_channels = 64

        self.conv1 = nn.Conv2d(input_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

        if self.dropin:
            self.layer2_original = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.in_channels = 64
            self.layer2_dropin = self._make_layer(block, 128, 1, stride=2)
            self.in_channels = 128 + 128
            self.layer3_original = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer3_dropin = self._make_layer(block, 256, 1, stride=2)
            self.in_channels = 256 + 256
        else:
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def freeze_original_layers(self):
        if self.dropin:
            for param in self.conv1.parameters(): param.requires_grad = False
            for param in self.bn1.parameters(): param.requires_grad = False
            for param in self.layer1.parameters(): param.requires_grad = False
            for param in self.layer2_original.parameters(): param.requires_grad = False
            for param in self.layer3_original.parameters(): param.requires_grad = False
            for param in self.layer4.parameters(): param.requires_grad = False
            for param in self.fc.parameters(): param.requires_grad = False
        else:
            print("No original layers to freeze (not in dropin mode)")

    def unfreeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        if self.dropin:
            original_params = (sum(p.numel() for p in self.conv1.parameters()) +
                               sum(p.numel() for p in self.bn1.parameters()) +
                               sum(p.numel() for p in self.layer1.parameters()) +
                               sum(p.numel() for p in self.layer2_original.parameters()) +
                               sum(p.numel() for p in self.layer3_original.parameters()) +
                               sum(p.numel() for p in self.layer4.parameters()) +
                               sum(p.numel() for p in self.fc.parameters()))
            dropin_params = (sum(p.numel() for p in self.layer2_dropin.parameters()) +
                             sum(p.numel() for p in self.layer3_dropin.parameters()))
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'frozen_params': frozen_params,
                'original_branch_params': original_params,
                'dropin_branch_params': dropin_params
            }
        else:
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'frozen_params': frozen_params
            }

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)

        if self.dropin:
            x1 = self.layer2_original(x)
            x2 = self.layer2_dropin(x)
            x = torch.cat([x1, x2], dim=1)
            x1 = self.layer3_original(x)
            x2 = self.layer3_dropin(x)
            x = torch.cat([x1, x2], dim=1)
        else:
            x = self.layer2(x)
            x = self.layer3(x)

        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18(input_c, dropin=False, num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], input_c, dropin, num_classes)

def test_model_shapes():
    print("Testing model shapes...")
    x = torch.randn(4, 1, 28, 28)
    model_baseline = ResNet18(input_c=1, dropin=False, num_classes=10)
    with torch.no_grad():
        out = model_baseline(x)
    print(f"Baseline output shape: {out.shape}")
    model_dropin = ResNet18(input_c=1, dropin=True, num_classes=10)
    with torch.no_grad():
        out = model_dropin(x)
    print(f"Dropin output shape: {out.shape}")
    model_dropin.freeze_original_layers()
    stats = model_dropin.get_param_stats()
    print("Stats:", stats)

if __name__ == "__main__":
    test_model_shapes()
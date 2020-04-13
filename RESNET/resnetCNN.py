import torch
import torchvision
import torch.nn as nn
from torchvision import models, transforms, datasets
import numpy as np
import torch.nn.functional as F

# Resnet18or34层基础块
class ResNet18or34BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResNet18or34BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, stride=stride, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannel, out_channels=outchannel, stride=stride, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        x = self.left(x)
        x += self.shortcut(x)
        x = F.relu(x)
        return x

# Resnet50or101or152层基础块
class ResNet50or101or152BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResNet50or101or152BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, stride=stride, kernel_size=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, stride=stride, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, stride=stride, kernel_size=1),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=inchannel, out_channels=outchannel, stride=1, kernel_size=1, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        x = self.left(x)
        x += self.shortcut(x)
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # resnet18
        self.layer1 = self.make_layer(ResNet18or34BasicBlock, 64, 2, 1)
        self.layer2 = self.make_layer(ResNet18or34BasicBlock, 128, 2, 2)
        self.layer3 = self.make_layer(ResNet18or34BasicBlock, 256, 2, 2)
        self.layer4 = self.make_layer(ResNet18or34BasicBlock, 512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

        '''
        resnet34
        self.layer1 = self.make_layer(ResNet18or34BasicBlock, 64, 3, 1)
        self.layer2 = self.make_layer(ResNet18or34BasicBlock, 128, 4, 2)
        self.layer3 = self.make_layer(ResNet18or34BasicBlock, 256, 6, 2)
        self.layer4 = self.make_layer(ResNet18or34BasicBlock, 512, 3, 2)
        self.fc = nn.Linear(512, num_classes)
        
        resnet50
        self.layer1 = self.make_layer(ResNet50or101or152BasicBlock, 256, 3, 1)
        self.layer2 = self.make_layer(ResNet50or101or152BasicBlock, 512, 4, 2)
        self.layer3 = self.make_layer(ResNet50or101or152BasicBlock, 1024, 6, 2)
        self.layer4 = self.make_layer(ResNet50or101or152BasicBlock, 2048, 3, 2)
        self.fc = nn.Linear(2048, num_classes)
        
        resnet101
        self.layer1 = self.make_layer(ResNet50or101or152BasicBlock, 256, 3, 1)
        self.layer2 = self.make_layer(ResNet50or101or152BasicBlock, 512, 4, 2)
        self.layer3 = self.make_layer(ResNet50or101or152BasicBlock, 1024, 23, 2)
        self.layer4 = self.make_layer(ResNet50or101or152BasicBlock, 2048, 3, 2)
        self.fc = nn.Linear(2048, num_classes)
        
        resnet152
        self.layer1 = self.make_layer(ResNet50or101or152BasicBlock, 256, 3, 1)
        self.layer2 = self.make_layer(ResNet50or101or152BasicBlock, 512, 8, 2)
        self.layer3 = self.make_layer(ResNet50or101or152BasicBlock, 1024, 36, 2)
        self.layer4 = self.make_layer(ResNet50or101or152BasicBlock, 2048, 3, 2)
        self.fc = nn.Linear(2048, num_classes)
        '''

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
            '''
            当resnet50，101，152时
            self.inchannel = self.inchannel * 2
            '''
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet18():

    """
       return ResNet(ResNet50or101or152BasicBlock)
    """
    return ResNet(ResNet18or34BasicBlock)
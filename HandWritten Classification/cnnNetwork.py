import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import SGD
from torch.autograd import Variable
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
from PIL import Image


'''定义CNN网络模型'''
class CNNetwork_norm(nn.Module):
    def __init__(self):
        super(CNNetwork_norm, self).__init__()

        # 定义2D 卷积层
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # 28 * 28 * 1   
        ''' C1有156个可训练参数（每个滤波器5*5=25个unit参数和一个bias参数，一共6个滤波器，共(5*5+1)*6=156个参数）'''
        self.conv1 = nn.Conv2d(1,6,5)   # n'= (n-f)/s+1
        self.norm1 = nn.BatchNorm2d(6)
        # 24 * 24 * 6 
        # MaxPool2d(kernelSize, stride)
        ''' 池化（Pooling）：也称为欠采样或下采样。主要用于特征降维，压缩数据和参数的数量，减小过拟合，同时提高模型的容错性。
            池化不需要参数控制 '''
        self.pool1 = nn.MaxPool2d(2, stride=2)   # n'= (n-f)/s+1
        # 12 * 12 * 6
        self.conv2 = nn.Conv2d(6,16,3)  # 原始28*28
        self.norm2 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(6,16,5)
        # 10 * 10 * 16
        self.pool2 = nn.MaxPool2d(2, 2)
        # 5 * 5 * 16
        self.fc1 = nn.Linear(5*5*16, 120)
        # self.fc1 = nn.Linear(10*10*16, 120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)
        # self.fc4 = nn.Linear(120,12)
        

    def forward(self, x):

        # layer1
        x = self.norm1(self.conv1(x))
        x = F.relu(x)
        x = self.pool1(x)

        # layer2
        x = self.norm2(self.conv2(x))
        x = F.relu(x)
        x =  self.pool2(x)

        # X.view(-1)中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的，并且按先后顺序排列。
        # x = x.view(x.size()[0], -1) //??
        # 将 x 转为一维向量形式，为全连接层做准备
        x = x.view(x.size()[0], -1)
        # x = x.view(-1, 16*5*5)
        # 全连接层 fc
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # 最后一层不过激活函数
        # d = self.fc4(x)
        # 为什么用relu函数？？
        return x


class CNNetwork_resize(nn.Module):
    def __init__(self):
        super(CNNetwork_resize, self).__init__()

        # 定义2D 卷积层
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # 32 * 32 * 1   
        ''' C1有156个可训练参数（每个滤波器5*5=25个unit参数和一个bias参数，一共6个滤波器，共(5*5+1)*6=156个参数）'''
        self.conv1 = nn.Conv2d(1,6,5)   # n'= (n-f)/s+1
        # 28 * 28 * 6 
        # MaxPool2d(kernelSize, stride)
        ''' 池化（Pooling）：也称为欠采样或下采样。主要用于特征降维，压缩数据和参数的数量，减小过拟合，同时提高模型的容错性。
            池化不需要参数控制 '''
        self.pool1 = nn.MaxPool2d(2, stride=2)   # n'= (n-f)/s+1
        # 14 * 14 * 6
        # self.conv2 = nn.Conv2d(6,16,3)  # 原始28*28
        self.conv2 = nn.Conv2d(6,16,5)
        # 10 * 10 * 16
        self.pool2 = nn.MaxPool2d(2, 2)
        # 5 * 5 * 16
        self.fc1 = nn.Linear(5*5*16, 120)
        # self.fc1 = nn.Linear(10*10*16, 120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)
        self.fc4 = nn.Linear(120,12)
        

    def forward(self, x):
        # layer1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # layer2
        x = self.conv2(x)
        x = F.relu(x)
        x =  self.pool2(x)

        # X.view(-1)中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的，并且按先后顺序排列。
        # x = x.view(x.size()[0], -1) //??
        # 将 x 转为一维向量形式，为全连接层做准备
        x = x.view(x.size()[0], -1)
        # x = x.view(-1, 16*5*5)
        # 全连接层 fc
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # 最后一层不过激活函数
        # d = self.fc4(x)
        # 为什么用relu函数？？
        return x


class CNNetwork(nn.Module):
    def __init__(self):
        super(CNNetwork, self).__init__()

        # 定义2D 卷积层
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # 28 * 28 * 1   
        ''' C1有156个可训练参数（每个滤波器5*5=25个unit参数和一个bias参数，一共6个滤波器，共(5*5+1)*6=156个参数）'''
        self.conv1 = nn.Conv2d(1,6,5)   # n'= (n-f)/s+1
        # 24 * 24 * 6 
        # MaxPool2d(kernelSize, stride)
        ''' 池化（Pooling）：也称为欠采样或下采样。主要用于特征降维，压缩数据和参数的数量，减小过拟合，同时提高模型的容错性。
            池化不需要参数控制 '''
        self.pool1 = nn.MaxPool2d(2, stride=2)   # n'= (n-f)/s+1
        # 12 * 12 * 6
        self.conv2 = nn.Conv2d(6,16,3)  # 原始28*28
        # self.conv2 = nn.Conv2d(6,16,5)
        # 10 * 10 * 16
        self.pool2 = nn.MaxPool2d(2, 2)
        # 5 * 5 * 16
        self.fc1 = nn.Linear(5*5*16, 120)
        # self.fc1 = nn.Linear(10*10*16, 120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)
        # self.fc4 = nn.Linear(120,12)
        

    def forward(self, x):
        # layer1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # layer2
        x = self.conv2(x)
        x = F.relu(x)
        x =  self.pool2(x)

        # X.view(-1)中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的，并且按先后顺序排列。
        # x = x.view(x.size()[0], -1) //??
        # 将 x 转为一维向量形式，为全连接层做准备
        x = x.view(x.size()[0], -1)
        # x = x.view(-1, 16*5*5)
        # 全连接层 fc
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # 最后一层不过激活函数
        # d = self.fc4(x)
        # 为什么用relu函数？？
        return x


class CNNetwork_32(nn.Module):
    def __init__(self):
        super(CNNetwork_32, self).__init__()

        # 定义2D 卷积层
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # 28 * 28 * 1   
        ''' C1有156个可训练参数（每个滤波器5*5=25个unit参数和一个bias参数，一共6个滤波器，共(5*5+1)*6=156个参数）'''
        self.conv1 = nn.Conv2d(1,6,5)   # n'= (n-f)/s+1
        # 24 * 24 * 6 
        # MaxPool2d(kernelSize, stride)
        ''' 池化（Pooling）：也称为欠采样或下采样。主要用于特征降维，压缩数据和参数的数量，减小过拟合，同时提高模型的容错性。
            池化不需要参数控制 '''
        self.pool1 = nn.MaxPool2d(2, stride=2)   # n'= (n-f)/s+1
        # 12 * 12 * 6
        self.conv2 = nn.Conv2d(6,32,3)  # 原始28*28
        # 10 * 10 * 16
        self.pool2 = nn.MaxPool2d(2, 2)
        # 5 * 5 * 16
        self.fc1 = nn.Linear(5*5*32, 120)
        # self.fc1 = nn.Linear(10*10*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)

    def forward(self, x):
        # layer1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # layer2
        x = self.conv2(x)
        x = F.relu(x)
        x =  self.pool2(x)

        # X.view(-1)中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的，并且按先后顺序排列。
        # x = x.view(x.size()[0], -1) //??
        # 将 x 转为一维向量形式，为全连接层做准备
        x = x.view(x.size()[0], -1)
        # x = x.view(-1, 16*5*5)
        # 全连接层 fc
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # 最后一层不过激活函数
        # d = self.fc4(x)
        # 为什么用relu函数？？
        return x


class CNNetwork_64(nn.Module):
    def __init__(self):
        super(CNNetwork_64, self).__init__()

        # 定义2D 卷积层
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # 28 * 28 * 1   
        ''' C1有156个可训练参数（每个滤波器5*5=25个unit参数和一个bias参数，一共6个滤波器，共(5*5+1)*6=156个参数）'''
        self.conv1 = nn.Conv2d(1,6,5)   # n'= (n-f)/s+1
        # 24 * 24 * 6 
        # MaxPool2d(kernelSize, stride)
        ''' 池化（Pooling）：也称为欠采样或下采样。主要用于特征降维，压缩数据和参数的数量，减小过拟合，同时提高模型的容错性。
            池化不需要参数控制 '''
        # self.pool1 = nn.MaxPool2d(4, stride=4)   # n'= (n-f)/s+1
        self.pool1 = nn.MaxPool2d(2, stride=2)   # n'= (n-f)/s+1
        # 12 * 12 * 6
        self.conv2 = nn.Conv2d(6,64,3)  # 原始28*28
        # self.conv2 = nn.Conv2d(6,16,5)
        # 10 * 10 * 16
        self.pool2 = nn.MaxPool2d(2, 2)
        # 6 * 6 * 16
        # self.fc1 = nn.Linear(6*6*6, 120)
        self.fc1 = nn.Linear(5*5*64, 120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)
        # self.fc4 = nn.Linear(120,12)
        

    def forward(self, x):
        # layer1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # layer2
        x = self.conv2(x)
        x = F.relu(x)
        x =  self.pool2(x)

        # X.view(-1)中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的，并且按先后顺序排列。
        # x = x.view(x.size()[0], -1) //??
        # 将 x 转为一维向量形式，为全连接层做准备
        x = x.view(x.size()[0], -1)
        # x = x.view(-1, 16*5*5)
        # 全连接层 fc
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # 最后一层不过激活函数
        # d = self.fc4(x)
        # 为什么用relu函数？？
        return x


class CNNetwork_oneFC(nn.Module):
    def __init__(self):
        super(CNNetwork_oneFC, self).__init__()

        # 定义2D 卷积层
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # 28 * 28 * 1   
        ''' C1有156个可训练参数（每个滤波器5*5=25个unit参数和一个bias参数，一共6个滤波器，共(5*5+1)*6=156个参数）'''
        self.conv1 = nn.Conv2d(1,6,5)   # n'= (n-f)/s+1
        # 24 * 24 * 6 
        # MaxPool2d(kernelSize, stride)
        ''' 池化（Pooling）：也称为欠采样或下采样。主要用于特征降维，压缩数据和参数的数量，减小过拟合，同时提高模型的容错性。
            池化不需要参数控制 '''
        self.pool1 = nn.MaxPool2d(2, stride=2)   # n'= (n-f)/s+1
        # 12 * 12 * 6
        self.conv2 = nn.Conv2d(6,16,3)  # 原始28*28
        # self.conv2 = nn.Conv2d(6,16,5)
        # 10 * 10 * 16
        self.pool2 = nn.MaxPool2d(2, 2)
        # 5 * 5 * 16
        self.fc1 = nn.Linear(5*5*16, 120)
        # self.fc1 = nn.Linear(10*10*16, 120)

        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 12)
        self.fc4 = nn.Linear(120,12)
    
        

    def forward(self, x):
        # layer1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # layer2
        x = self.conv2(x)
        x = F.relu(x)
        x =  self.pool2(x)

        # X.view(-1)中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的，并且按先后顺序排列。
        # x = x.view(x.size()[0], -1) //??
        # 将 x 转为一维向量形式，为全连接层做准备
        x = x.view(x.size()[0], -1)
        # x = x.view(-1, 16*5*5)
        # 全连接层 fc
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x) # 最后一层不过激活函数
        d = self.fc4(x)
        # 为什么用relu函数？？
        return x


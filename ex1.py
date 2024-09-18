import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt

#训练超参数设置，构建训练数据
# Hyper prameters
EPOCH = 2
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    download=DOWNLOAD_MNIST
)

print(train_data.data.size())
print(train_data.targets.size())
print(train_data.data[0])


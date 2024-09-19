import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# 画一个图片显示出来
plt.imshow(train_data.data[0].numpy(),cmap='gray')
plt.title('%i'%train_data.targets[0])
plt.show()

# 训练和测试数据准备
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
)

# 这里只取前3千个数据吧，差不多已经够用了，然后将其归一化。
with torch.no_grad():
    test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:3000] / 255
    test_y = test_data.targets[:3000]

'''开始建立CNN网络'''


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        '''
        一般来说，卷积网络包括以下内容：
        1.卷积层
        2.神经网络
        3.池化层
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # --> (1,28,28)
                in_channels=1,  # 传入的图片是几层的，灰色为1层，RGB为三层
                out_channels=16,  # 输出的图片是几层
                kernel_size=5,  # 代表扫描的区域点为5*5
                stride=1,  # 就是每隔多少步跳一下
                padding=2,  # 边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=2
            ),  # 2d代表二维卷积           --> (16,28,28)
            nn.ReLU(),  # 非线性激活层
            nn.MaxPool2d(kernel_size=2),  # 设定这里的扫描区域为2*2，且取出该2*2中的最大值          --> (16,14,14)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(  # --> (16,14,14)
                in_channels=16,  # 这里的输入是上层的输出为16层
                out_channels=32,  # 在这里我们需要将其输出为32层
                kernel_size=5,  # 代表扫描的区域点为5*5
                stride=1,  # 就是每隔多少步跳一下
                padding=2,  # 边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=
            ),  # --> (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 设定这里的扫描区域为2*2，且取出该2*2中的最大值     --> (32,7,7)，这里是三维数据
        )

        self.out = nn.Linear(32 * 7 * 7, 10)  # 注意一下这里的数据是二维的数据

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # （batch,32,7,7）
        # 然后接下来进行一下扩展展平的操作，将三维数据转为二维的数据
        x = x.view(x.size(0), -1)  # (batch ,32 * 7 * 7)
        output = self.out(x)
        return output


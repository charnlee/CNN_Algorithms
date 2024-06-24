import torch
from torch import nn
from torchsummary import summary

#定义LeNet模型类
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        #一般卷积层增加通道数，池化层减少单个数据的维度
        #定义第一个卷积层的配置，输入为数据以1 * 28 * 28为例， 卷积之后6 * 28 * 28
        self.c1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        #定义sigmoid激活函数
        self.sig = nn.Sigmoid()
        #定义第一个平均池化层，池化之后变为6 * 14 *14
        self.s2 = nn.AvgPool2d(kernel_size=2,stride=2)
        #定义第二个卷积层的配置，由一下卷积之后变为16 * 10 *10
        self.c3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        #定义第二个平均池化层
        self.s4 = nn.AvgPool2d(kernel_size=2,stride=2)
        #将得到的最终结果展成一维数据
        self.flatten = nn.Flatten()
        #将此时的一维数据经过全连接运算，把数据压缩到120维
        self.f5 = nn.Linear(5 * 5 * 16, 120)
        #将此时的120维数据压缩到84维
        self.f6 = nn.Linear(120, 84)
        #将84维的数据压缩到10维
        self.f7 = nn.Linear(84, 10)

    def forward(self,x):
        #调用卷积和激活函数
        x = self.sig(self.c1(x))
        #卷积过后调用池化
        x = self.s2(x)
        #调用卷积和激活函数
        x = self.sig(self.c3(x))
        #卷积过后再池化
        x = self.s2(x)
        #下面三层全连接层
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LeNet().to(device)
    print(summary(model,(1,28,28)))

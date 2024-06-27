import torch
from torch import nn
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,use_conv=False,stride=1):
        super().__init__()
        self.re = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=stride)
        self.c2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_conv:
            self.c3 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=1,stride=stride)
        else:
            self.c3 = None

    def forward(self,x):
        y = self.re(self.bn1(self.c1(x)))
        y = self.bn2(self.c2(y))
        if self.c3:
            x = self.re(y)
        y = self.re(y+x)
        if self.c3:
            x = self.re(y)
        return y

class ResNet18(nn.Module):
    def __init__(self,Residual):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.b2 = nn.Sequential(
            Residual(64,64,use_conv=False,stride=1),
            Residual(64,64,use_conv=False,stride=1)
        )
        self.b3 = nn.Sequential(
            Residual(64,128,use_conv=True,stride=2),
            Residual(128,128,use_conv=False,stride=1)
        )
        self.b4 = nn.Sequential(
            Residual(128,256,use_conv=True,stride=2),
            Residual(256,256,use_conv=False,stride=1)
        )
        self.b5 = nn.Sequential(
            Residual(256,512,use_conv=True,stride=2),
            Residual(512,512,use_conv=False,stride=1)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(Residual).to(device)
    print(summary(model,(1,224,224)))
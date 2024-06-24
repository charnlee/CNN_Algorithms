import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.ReLU = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4)
        self.p1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.c2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        self.p2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.c3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
        self.c4 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1)
        self.c5 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1)
        self.p3 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.f = nn.Flatten()
        self.f1 = nn.Linear(9216,4096)
        self.f2 = nn.Linear(4096,4096)
        self.f3 = nn.Linear(4096,10)

    def forward(self,x):
        x = self.ReLU(self.c1(x))
        x = self.p1(x)
        x = self.ReLU(self.c2(x))
        x = self.p2(x)
        x = self.ReLU(self.c3(x))
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.p3(x)
        x = self.f(x)
        x = self.ReLU(self.f1(x))
        x = F.dropout(x,0.5)
        x = self.ReLU(self.f2(x))
        x = F.dropout(x,0.5)
        x = self.f3(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
    model = AlexNet()
    model = model.to(device)

    print(summary(model,(1,227,227)))


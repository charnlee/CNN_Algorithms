import copy
import time

from torchvision.datasets import FashionMNIST
import torch
from torch import nn
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import AlexNet
import pandas as pd

#数据集加载和处理
def train_val_data_process():
    #数据下载
    train_data = FashionMNIST(root="./data",
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                              download=True)
    # 数据集划分
    train_data,val_data = Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    #加载训练数据到数据容器
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=0)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=0)
    return train_dataloader,val_dataloader


# 模型训练
def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda")

    #创建优化器对象
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    #定义损失函数对象
    criterion = nn.CrossEntropyLoss()
    #加载模型到GPU
    model = model.to(device)
    #记录模型参数
    best_model_wts = copy.deepcopy(model.state_dict())
    #记录最佳模型
    best_acc = 0.0
    #创建所有训练数据的损失函数列表变量
    train_loss_all = []
    #创建所有验证数据的损失函数列表变相
    val_loss_all = []
    #创建整体训练数据准确度列表
    train_acc_all = []
    #创建整体验证数据准确度列表
    val_acc_all = []
    #记录时间
    since = time.time()

    # 一轮一轮训练数据
    for epoch in range(num_epochs):
        print(f"第 {epoch+1}/{num_epochs} 轮训练开始,使用{device}进行训练")
        print("-"*20)

        train_loss = 0.0
        train_corrects = 0

        val_loss = 0.0
        val_corrects = 0

        train_num = 0
        val_num = 0
        #对于每一个批次的数据
        for step,(b_x,b_y) in enumerate(train_dataloader):
            #将张量放到GPU上
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #开启训练模模式
            model.train()
            #训练模型
            output = model(b_x)
            #记录最大的输出结果
            pre_label = torch.argmax(output, dim=1)
            #计算损失
            loss = criterion(output, b_y)
            #梯度清零
            optimizer.zero_grad()
            #计算用来梯度更新的值
            loss.backward()
            #计算梯度更新后的梯度
            optimizer.step()
            #所有批次的了损失总和
            train_loss+= loss.item() * b_x.size(0)
            #累加预测正确的数量
            train_corrects += torch.sum(pre_label == b_y.data)
            #累加所有的训练样本的数量
            train_num += b_x.size(0)
        # 取数据加载器里面的数据
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #开启评估模式
            model.eval()
            #计算前向传播的值
            output = model(b_x)
            #选取预测结果的值
            pre_lab = torch.argmax(output, dim=1)
            # 计算损失
            loss = criterion(output, b_y)
            # 累加每个批次的损失
            val_loss += loss.item() * b_x.size(0)
            #计算预测正确的所有样本数量
            val_corrects += torch.sum(pre_lab==b_y.data)
            # 计算总样本的数量
            val_num += b_x.size(0)
        # 把每一轮的损失加到损失列表里
        train_loss_all.append(train_loss / train_num)
        # 把每一轮的正确率加到正确率列表里
        train_acc_all.append(train_corrects.double().item() / train_num)


        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print(f"第 {epoch+1} 轮 Train Loss: {train_loss_all[-1]:.4f} Train Acc: {train_acc_all[-1]:.4f}")
        print(f"第 {epoch+1} 轮 Val Loss: {val_loss_all[-1]:.4f} Val Acc:{val_acc_all[-1]:.4f}")

        #找出最优的准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print(f"训练耗费时间为：{time_use//60:.0f}m{time_use%60:.0f}s")

    # 选择最优参数
    # 加载最高准确率下的模型参数
    # model.load_state_dict(best_model_wts)
    torch.save(best_model_wts,'../AlexNet/best_model.pth')


    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all})
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label="train loss")
    plt.plot(train_process["epoch"],train_process.val_loss_all,'bs-',label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"],train_process.train_acc_all,'ro-',label="train loss")
    plt.plot(train_process["epoch"],train_process.val_acc_all,'bs-',label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

if __name__ == "__main__":
    lenet = AlexNet()
    train_dataloader,val_dataloader = train_val_data_process()
    train_process = train_model_process(lenet,train_dataloader,val_dataloader,20)
    matplot_acc_loss(train_process)









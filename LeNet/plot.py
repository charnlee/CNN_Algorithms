from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

# 下载处理数据集FashionMNIST
train_data = FashionMNIST(root="./data",
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                          download=True)

# 加载数据集里的数据到容器中
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

#  从数据容器中取第一个数据
for step,(b_x,b_y) in enumerate(train_loader):
    if step > 0:
        break

    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
# print(step,b_x.shape,b_y.shape)
# print(step,batch_x.shape,batch_y.shape)
    class_label = train_data.classes
    print(class_label)

plt.figure(figsize=(12,5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4,16,ii+1)
    plt.imshow(batch_x[ii, :, :],cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]],size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()



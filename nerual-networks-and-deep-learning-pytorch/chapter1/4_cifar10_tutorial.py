"""训练一个分类器
https://github.com/zergtant/pytorch-handbook/blob/master/chapter1/4_cifar10_tutorial.ipynb
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# gets the absolute path to the file
import os
import matplotlib.pyplot as plt
import numpy as np
import time
abspath = os.path.abspath(os.path.dirname(__file__))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 206, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(206, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def downloadCIFAR10():
    """download CIFAR 10 data
    """
    # nomolize mean and varance from (0,1) to (-1,1), -1=(0 - 0.5)/0.5, 1=(1 - 0.5)/0.5
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root=abspath + '/data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=abspath + '/data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

def imshow(img):
    """展示图像的函数
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def showSomeData(trainloader, testloader, classes):
    """download CIFAR 10 data
    """

    # 获取随机数据
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # 展示图像
    imshow(torchvision.utils.make_grid(images))
    # 显示图像标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



if __name__ == "__main__":
    useGPU = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### 1.load data
    trainloader, testloader, classes = downloadCIFAR10() 
    # showSomeData(trainloader, testloader, classes), plt.show()

    ### 2. create net
    net = Net()
    if useGPU:
        net.to(device)

    # cross entropy, learning rate and momentum
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    ### 3.training
    for epoch in range(2):  # 多批次循环

        running_loss = 0.0
        time_start = time.time()
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data
            if useGPU:
                inputs, labels = inputs.to(device), labels.to(device)

            # 梯度置0
            optimizer.zero_grad()

            # 正向传播，反向传播，优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            running_loss += loss.item()
            if i % 2000 == 1999:    # 每2000批次打印一次
                print('[%d, %5d] loss: %.3f, time: %.5f' %
                    (epoch + 1, i + 1, running_loss / 2000, time.time() - time_start))
                time_start = time.time()
                running_loss = 0.0

    print('Finished Training')


    ### 4.test
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            if useGPU:
                inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1) # max of every row
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item() # num + true = num+1
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s: %2d%%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

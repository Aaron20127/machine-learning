"""神经网络
https://github.com/zergtant/pytorch-handbook/blob/master/chapter1/3_neural_networks_tutorial.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window, stride == filter_size
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number, stride == filter_size
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x)) # note x is a row vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


### 1.all layers
print("\n>>>> 1.all layers >>>>")
net = Net()
print(net)

### 2.wights and bias of every layers
print("\n>>>> 2.wights and bias of every layers >>>>")
params = list(net.parameters())
print(len(params))

print(params[0].size())  # conv1's .weight, namely filter core
print(params[1].size())  # conv1's .bais

print(params[2].size())  # conv2's .weight, namely filter core
print(params[3].size())  # conv2's .bais

print(params[4].size())  # fc1's .weight
print(params[5].size())  # fc1's .bais

print(params[6].size())  # fc2's .weight
print(params[7].size())  # fc2's .bais

print(params[8].size())  # fc3's .weight
print(params[9].size())  # fc3's .bais


### 3.forword and update grad
print("\n>>>> 3.forword and update grad >>>>")
input = torch.randn(1, 1, 32, 32)  # sSamples * nChannels * Height * Width
output = net(input) # forword
lable = torch.randn(1,10) # create random lable

# create loss
criterion = nn.MSELoss()
loss = criterion(output, lable)
net.zero_grad() 

# backward ans update grad
optimizer = optim.SGD(net.parameters(), lr=0.01) # # create your optimizer, lr - learning rate
optimizer.zero_grad() # zero the gradient buffers

print("\nBefore, net.conv1.bias.grad: \n", net.conv1.bias.grad)
print("\nBefore, net.conv1.bias: \n", net.conv1.bias)
loss.backward(retain_graph=True)
optimizer.step() # update grad
print("\nAfter, net.conv1.bias.grad: \n", net.conv1.bias.grad)
print("\nAfter, net.conv1.bias: \n", net.conv1.bias)
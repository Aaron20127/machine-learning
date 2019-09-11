# 引用
# 注意，这里我们使用了一个新库叫 seaborn 如果报错找不到包的话请使用pip install seaborn 来进行安装
import torch
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
torch.__version__

x = np.linspace(0,20,500)
y = 5*x + 7
plt.figure()
plt.plot(x,y)


x = np.random.rand(256)
noise = np.random.randn(256) / 4
y = x * 5 + 7 + noise

# 绘图
# df = pd.DataFrame()
# df['x'] = x
# df['y'] = y
# sns.lmplot(x='x', y='y', data=df);


model=Linear(1, 1)
criterion = MSELoss()
# criterion = torch.nn.L1Loss()
# criterion = torch.nn.CrossEntropyLoss

optim = SGD(model.parameters(), lr = 0.01)
# optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optim = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
# optim = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08)

epochs = 6000

x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')


for i in range(epochs):
    # 整理输入和输出的数据，这里输入和输出一定要是torch的Tensor类型
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    #使用模型进行预测
    outputs = model(inputs)
    #梯度置0，否则会累加
    optim.zero_grad()
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 使用优化器默认方法优化
    optim.step()
    if (i%100==0):
        #每 100次打印一下损失函数，看看效果
        print('epoch {}, loss {:1.4f}'.format(i,loss.data.item()))

# 训练得到的参数
[w, b] = model.parameters()
print (w.item(),b.item())
# 偏差
print (5-w.item(),7-b.item())

predicted = model.forward(torch.from_numpy(x_train)).data.numpy()

plt.figure()
plt.plot(x_train, y_train, 'go', label = 'data', alpha = 0.3)
plt.plot(x_train, predicted, label = 'predicted', alpha = 1)
plt.legend()
plt.show()
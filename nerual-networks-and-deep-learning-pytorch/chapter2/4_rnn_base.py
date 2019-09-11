"""RNN 基本原理示例
https://www.youtube.com/watch?v=ogZi5oIo4fI
"""

import torch
import torch.nn as nn
import numpy as np
import sys

# # RNN测试, hn和h0相同，output和input只有最后个维度不相同
# rnn = torch.nn.RNN(input_size=4, hidden_size=2, batch_first=True)
# input = torch.randn(3, 5, 4) # batch, sequnce, input_size
# h_0 =torch.randn(1, 3, 2) #  ~, batch, hidden_size
# output,hn=rnn(input ,h_0) # output = (batch, sequnce, hidden_size), hn(~, batch, hidden_size)
# print(output.size(),hn.size())


# 翻译 hihell -> ihello
### 1.prepare data
idx2char = ['h', 'i', 'e', 'l', 'o']

x_data = [0, 1, 0, 2, 3, 3]  # hihell
y_data = [1, 0, 2, 3, 3, 4]  # ihello

one_hot_lookup = np.eye(5)   # 每一行为一个字母编码序列
inputs_np = np.array([one_hot_lookup[x] for x in x_data]) # 输入编码

inputs = torch.from_numpy(inputs_np).double()
labels = torch.Tensor(y_data).long() # 由于是分类，标签是整形，特别是交叉熵的标签


### 2.parameters
num_classes = 5  # 字母的类别
input_size = 5   # 输入的每个字母的编码长度
hidden_size = 5  # 输出字母的编码长度
batch_size = 1   # 一次训练的样本个数
sequence_length = 6 # 序列的个数，这里一个单词由6个字符组成
num_layers = 1  # 这个暂时不清楚


### 3. Model
class Model(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        # input_size 每个节点的输入元素个数
        # hidden_size 每个节点的输出元素个数
        self.rnn = nn.RNN(input_size = input_size, \
                          hidden_size = hidden_size, \
                          batch_first = True)

    def forward(self, x, hidden):
        # Reshape input in (batch_size, sequence_length, input_size)
        x = x.view(batch_size, sequence_length, input_size)

        # forward
        # input: (batch, seq_len, input_size)
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, num_classes) # 每行一个节点元素
        return out

    def init_hidden(self):
        # Initalize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        return torch.zeros([num_layers, batch_size, hidden_size], dtype=torch.double)


### 4. training
model = Model(input_size, hidden_size).double()

# 注意交叉熵的criterion的第一个参数shape(m,n),每一行表示一个样本的输出，double类型
# 第二个参数shape(m,)，是第一个参数每一行的标签，long类型，整形的0或1
criterion = torch.nn.CrossEntropyLoss() # Logsoftmax + NLLLoss 
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    hidden = model.init_hidden() # 起始隐藏层
    outputs = model(inputs, hidden) # forward整个RNN
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(1) # 得到预测字母编号
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx] # 预测字母
    print("epoch: %d, Predicted string: %s, loss: %1.3f" % \
          (epoch+1, ''.join(result_str), loss.item())) 

print("Learning finished!")
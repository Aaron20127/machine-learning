"""Autograd：自动求导
https://github.com/zergtant/pytorch-handbook/blob/master/chapter1/2_autograd_tutorial.ipynb
"""


from __future__ import print_function
import torch

# 1. y.backward(w) 等价于 torch.sum(y*w) 对x求导数
#
print("\n>>>>test.1>>>>")

x = torch.tensor([1.,2.], requires_grad=True)
w = torch.tensor([1.,2.])
y=x

y.backward(w)
print('y.backward(w): ',x.grad)

#
x = torch.tensor([1.,2.], requires_grad=True)
w = torch.tensor([1.,2.])
y=x

z = torch.sum(y*w) 
z.backward()
print('z.backward(): ', x.grad)

# 2. 
# 使用 with torch.no_grad()
# 计算过程中新生成的tensor不计算梯度，
# 而已经存在的tensor任然计算梯度
print("\n>>>>test.2>>>>")

x = torch.tensor([1.,2.], requires_grad=True)

with torch.no_grad():
    print(x.requires_grad)
    print((x ** 2).requires_grad)

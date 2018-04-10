#!/usr/bin/python
# -*- coding: UTF8 -*-

import numpy as np

def h_print(string):
    print '\n--------------------------------------------'
    print string
    print '--------------------------------------------'

def c_print(count):
    print count
    print '--------------------------------------------'

# 3维矩阵转换成二维矩阵
h_print('3维矩阵转换成二维矩阵')
b = np.array([[[1],[2],[3]],[[4],[5],[6]]])

c_print (b)
c_print (b.reshape(2,3))
c_print (b.reshape(2,3).transpose())

# 使矩阵矩阵的列向量相加成一列，n*m乘上m*1全为1的矩阵，需要列相加就消除掉列
# 使矩阵矩阵的列向量相加成一行，1*n全为1的矩阵乘上n*m，需要行相加就消除掉行
h_print('使矩阵矩阵的列向量相加成1列，或行向量加成1行')
a = np.array([[1,1],[2,2],[3,3]])
b = np.ones((a.shape[1],1))
c = np.ones((1,a.shape[0]))

c_print (a)
c_print (b)
c_print (c)
c_print (np.dot(a,b)) # 列相加
c_print (np.dot(c,a)) # 行相加

# 生成一个任意维数矩阵，元素是等差数列或者是相同的值
h_print('生成一个任意维数矩阵，元素是等差数列')
a = np.linspace(-1, 1, 10, endpoint=False)
b = np.linspace(-1, 1, 10, endpoint=False).reshape(2,5) # 重新生成2*5的矩阵
c = np.ones((2,3)) * 0.5 # 生成 2*3的元素为0.5的矩阵

c_print (a)
c_print (b)
c_print (c)
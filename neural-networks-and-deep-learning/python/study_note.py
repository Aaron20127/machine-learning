#!/usr/bin/python
# -*- coding: UTF8 -*-

import matplotlib.pyplot as plt
import numpy as np

def h_print(string):
    print '\n--------------------------------------------'
    print string
    print '--------------------------------------------'

def c_print(count):
    print count
    print '--------------------------------------------'


#### matrix 矩阵的一些测试
class matrixTest(object):

    def test_1(self):
        """ 3维矩阵转换成二维矩阵
        """
        h_print('3维矩阵转换成二维矩阵')
        b = np.array([[[1],[2],[3]],[[4],[5],[6]]])

        c_print (b)
        c_print (b.reshape(2,3))
        c_print (b.reshape(2,3).transpose())

    def test_2(self):
        """ 使矩阵矩阵的列向量相加成一列，n*m乘上m*1全为1的矩阵，需要列相加就消除掉列
            使矩阵矩阵的列向量相加成一行，1*n全为1的矩阵乘上n*m，需要行相加就消除掉行
        """
        h_print('使矩阵矩阵的列向量相加成1列，或行向量加成1行')
        a = np.array([[1,1],[2,2],[3,3]])
        b = np.ones((a.shape[1],1))
        c = np.ones((1,a.shape[0]))

        c_print (a)
        c_print (b)
        c_print (c)
        c_print (np.dot(a,b)) # 列相加
        c_print (np.dot(c,a)) # 行相加

    def test_3(self):
        """ 生成一个任意维数矩阵，元素是等差数列或者是相同的值
        """
        h_print('生成一个任意维数矩阵，元素是等差数列')
        a = np.linspace(-1, 1, 10, endpoint=False)
        b = np.linspace(-1, 1, 10, endpoint=False).reshape(2,5) # 重新生成2*5的矩阵
        c = np.ones((2,3)) * 0.5 # 生成 2*3的元素为0.5的矩阵

        c_print (a)
        c_print (b)
        c_print (c)

    def test_4(self):
        print self
        print self.__class__

#### 画图测试 https://blog.csdn.net/qq_31192383/article/details/53977822
class plotTest:

    def test_1(self):
        """一幅图片画三条曲线
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        y_1 = [7,10,7,10,7,10] # y坐标，x和y坐标都是向量，且长度都相等
        x_1 = np.arange(0, len(y_1)) # x坐标

        y_2 = [2,5,2,5,2,5] # y坐标，x和y坐标都是向量，且长度都相等
        x_2 = np.arange(0, len(y_2)) # x坐标

        # 16进制RGB http://www.114la.com/other/rgb.htm
        # linewith 是线宽
        # label 是线的名称
        ax.plot(x_1, y_1, color='#FFA933', linewidth=2.0,
                label="line 1")

        ax.plot(x_2, y_2, color='#FF4040', linewidth=2.0,
                label="line 2")        

        ax.set_xlim([0, 8]) # x坐标显示的宽度
        ax.set_xlabel('Epoch') # x坐标的意义
        ax.set_ylim([0, 15]) # y坐标的宽度
        ax.set_title('Classification accuracy') # 标题
        plt.legend(loc="lower right") # 线条的名称显示在右下角
        plt.show()

plotTest().test_1() 

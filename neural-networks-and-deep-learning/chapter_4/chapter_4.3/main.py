#!/usr/bin/python
# -*- coding: UTF8 -*-

"""
本节使用3层神经网络，模拟双输入和单输出的三维函数
"""

import sys
sys.path.append("../../common")

import random
import plot_figure
import network2
import matplotlib.pyplot as plt
import numpy as np

def test_0():
    """两层神经元，两个输入神经元x,y，一个输出神经元z。
    1.z在y的权重w2=0，z在x的权重w1=1000
    2.将输出阶跃点设置在0.5处，b = -0.5*w1 = -500
    3.以x，y为输入，画出z = f(x,y)的3d图形
    """

    def test(w1, w2, s):
        def fun(x, y):
            b = -s*w1
            return network2.sigmoid(x*w1 + y*w2 + b)

        x = y = np.linspace(0, 1, 100, endpoint=False)

        plot_figure.plot_base_3d(
            y_coordinate = y,
            x_coordinate= x,
            z_function = fun,
            x_lable='X',
            y_lable='Y',
            z_lable='Sigmod(x*w1 + y*w2 + b)') 
   
    test(10, 0, 0.5)
    test(100, 0, 0.5)
    test(1000, 0, 0.5)
    plt.show()

def test_1():
    """3层神经元，两个输入神经元x,y，两个隐藏神经元z1,z2，一个输出神经元z。
    1.z关于z1和z2的权重为h和-h
    2.y到隐藏层神经元的权重为0，x到隐藏层神经元的权重为都为1000，隐藏层的阶跃点分别为0.3和0.7
    3.输出层的偏置为0
    3.以x，y为输入，画出输出层的输入（w1*a1 + w2*a2 + b）的3d图形
    """

    def test(Wx1, Wy1, s1, Wx2, Wy2, s2, h):
        def fun(x, y):

            # 隐藏层
            if Wx1 and Wx2:
                b1 = -s1*Wx1 
                b2 = -s2*Wx2
            if Wy1 and Wy2:
                b1 = -s1*Wy1 
                b2 = -s2*Wy2

            a1 = network2.sigmoid(x*Wx1 + y*Wy1 + b1)
            a2 = network2.sigmoid(x*Wx2 + y*Wy2 + b2)

            # 输出层
            w1 = h
            w2 = -w1
            b = 0

            return w1*a1 + w2*a2 + b

        x = y = np.linspace(0, 1, 100, endpoint=False)

        plot_figure.plot_base_3d(
            y_coordinate = y,
            x_coordinate= x,
            z_function = fun,
            x_lable='X',
            y_lable='Y',
            z_lable='w1*a1 + w2*a2 + b') 
   
    # x轴方向凸
    test(1000, 0, 0.3, 1000, 0, 0.7, 0.6)
    # x轴方向凹
    test(1000, 0, 0.7, 1000, 0, 0.3, 0.6)

    # y轴方向凸
    test(0, 1000, 0.3, 0, 1000, 0.7, 0.6)
    # y轴方向凹
    test(0, 1000, 0.7, 0, 1000, 0.3, 0.6)

    plt.show()


def test_2():
    """3层神经元，两个输入神经元x,y，两对隐藏神经元z1,z2,z3,z4，一个输出神经元z。
    1.z关于z1和z2的权重为h和-h，z3和z4分别为h和-hs
    2.y到隐藏层神经元的权重为0，x到隐藏层神经元的权重为都为1000，隐藏层的阶跃点分别为0.3和0.7
    3.输出层的偏置为0
    3.以x，y为输入，画出输出层的输出z = sigmod（w1*a1 + w2*a2 + w3*a3 + w4*a4 + b）的3d图形
    """
    def test(Wx1, Wy1, s1, Wx2, Wy2, s2, Wx3, Wy3, s3, Wx4, Wy4, s4, h, b):
        def fun(x, y):

            # 隐藏层
            if Wx1 and Wx2:
                b1 = -s1*Wx1 
                b2 = -s2*Wx2
            if Wy1 and Wy2:
                b1 = -s1*Wy1 
                b2 = -s2*Wy2

            if Wx3 and Wx3:
                b3 = -s3*Wx3 
                b4 = -s4*Wx4
            if Wy4 and Wy4:
                b3 = -s3*Wy3 
                b4 = -s4*Wy4

            a1 = network2.sigmoid(x*Wx1 + y*Wy1 + b1)
            a2 = network2.sigmoid(x*Wx2 + y*Wy2 + b2)
            a3 = network2.sigmoid(x*Wx3 + y*Wy3 + b3)
            a4 = network2.sigmoid(x*Wx4 + y*Wy4 + b4)

            # 输出层
            w1 = w3 = h
            w2 = w4 = -w1

            return network2.sigmoid(w1*a1 + w2*a2 +  w3*a3 + w4*a4 + b)

        x = y = np.linspace(0, 1, 100, endpoint=False)

        plot_figure.plot_base_3d(
            y_coordinate = y,
            x_coordinate= x,
            z_function = fun,
            x_lable='X',
            y_lable='Y',
            z_lable='sigmod (z)') 


    # 设置输出层的h >= 10，b = -3/2 * h时，塔的底层消失，只有一个凸出的位置
    test(1000, 0, 0.5, 
         1000, 0, 0.6, 
         0, 1000, 0.5, 
         0, 1000, 0.6, h = 1, b = 0)

    test(1000, 0, 0.5, 
         1000, 0, 0.6, 
         0, 1000, 0.5, 
         0, 1000, 0.6, h = 10, b = -3/2*10)

    test(1000, 0, 0.5, 
         1000, 0, 0.6, 
         0, 1000, 0.5, 
         0, 1000, 0.6, h = 100, b = -3/2*100)

    test(1000, 0, 0.5, 
         1000, 0, 0.6, 
         0, 1000, 0.5, 
         0, 1000, 0.6, h = 1000, b = -3/2*1000)

    plt.show()

test_2()

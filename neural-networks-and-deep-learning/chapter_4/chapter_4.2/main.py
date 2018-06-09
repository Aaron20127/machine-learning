#!/usr/bin/python
# -*- coding: UTF8 -*-

"""
本节使用3层神经网络，模拟单输入和单输出的二维函数
"""

import sys
sys.path.append("../../common")

import random
import plot_figure
import network2
import matplotlib.pyplot as plt
import numpy as np

def get_sigmod_coordinate(w, b):
    x = np.linspace(0, 1, 10000, endpoint=False) # 输入从0-1，一共10000个点
    z = w*x + b
    y = network2.sigmoid(z)
    return x, y

def pair_hidden_neurons_out_put(s1, s2, h):
    """两个隐藏神经元的加权输出坐标
    """
    # 隐藏层
    w11 = w12 = 1000
    s11 = s1; s12 = s2
    b11 = - s11 * w11
    b12 = - s12 * w12 
    x, a1 = get_sigmod_coordinate(w11, b11)
    x, a2 = get_sigmod_coordinate(w12, b12) 

    # 输出层
    w1 = h
    w2 = -w1
    b = 0

    z = w1*a1 + w2*a2

    return x, z

def output_input(s1, s2, h):
    x, y = pair_hidden_neurons_out_put(s1, s2, h)

    plot_figure.plot_base(
        y_coordinate = [y],
        x_coordinate= [x],
        title='s1 = %.2f, s2 = %.2f, h = %.2f' % (s1, s2, h),
        x_lable='X',
        y_lable='w1*a1 + w2*a2',
        x_limit = [min(x)-0.2, max(x)+0.2],
        y_limit = [min(y)-0.2, max(y)+0.2]) 

def test_0():
    """单个神经元权重w与神经元输出的关系，且存在sigmod的阶跃点s = -b/w关系，
    如b = -1, w = 2，阶跃点就在0.5处。w越大，输出就越接近阶跃函数
    """
    for w in np.linspace(0, 100, 10, endpoint=False):
        b = -0.5 * w  # 阶跃点为0.5时求得的偏置的值
        x, y = get_sigmod_coordinate(w, b)
        plot_figure.plot_base(
            y_coordinate = [y],
            x_coordinate= [x],
            x_lable='x',
            y_lable='Sigmod (z)') 
    
    plt.show()

def test_1():
    """二维情况，有3层网络，输入层一个神经元，隐藏层两个神经元，输出层一个神经元。
    1.将隐藏层的权重设置成w = 1000，将阶跃点设置成0.4和0.6
    2.输出层的偏置设置为0，输出层的权重w1=-w2=h，表示了输出的高度
    3.画出隐藏层输出的加权和，即输出层的输入z，是一个凸的函数
    4.当第一个隐藏层的阶跃点超过0.6时，变成了凹函数
    """

    output_input(0.4, 0.6, 1) #凸起阶跃函数
    output_input(0.5, 0.6, 1) #凸起阶跃函数，凸起减小
    output_input(0.6, 0.6, 1) #输出为0
    output_input(0.7, 0.6, 1) #凹阶跃函数
    output_input(0.8, 0.6, 1) #凹阶跃函数，凹处增加
     
    plt.show()

def test_2():
    """二维情况，有3层网络，输出层一个神经元，隐藏层n对神经元，每对两个神经元组成，即0-1被分成了1/n个区间,
    每个区间可以是凸起和凹下表示，则n越大则输出越近似一条连续的曲线。输出层为一个神经元。
    1.y在-1到1之间，x在0-1之间，中间分成若干区间，随机拟合函数 f(x) = 0.2 + 0.4*x*x + 0.3*x*np.sin(15*x) + 0.05*np.cos(50*x)
    2.注意，最后是用输出层的输入z拟合的f(x)，并不是输出层最终的输出
    """
    # 拟合的函数
    def fitting_functions(x):
        # return random.uniform(-1,1) # 随机拟合函数
        return 0.2 + 0.4*x*x + 0.3*x*np.sin(15*x) + 0.05*np.cos(50*x)

    def n_random_pair_hidden_neurons_out_put(n, fun):
        """画出n个对隐藏层神经元的加权输出,fun是需要拟合的单输入和单输出的函数
        """
        y = None
        x = None

        for i in np.linspace(0, 1, n, endpoint=False):
            s1 = i; s2 = i+1.0/n; 
            h = fun(i) # 区间高度
            x, y1 = pair_hidden_neurons_out_put(s1, s2, h)

            if y == None:
                y = y1
            else:
                y += y1

        plot_figure.plot_base(
            y_coordinate = [y],
            x_coordinate= [x],
            title='n = %d' % (n),
            x_lable='X',
            y_lable='w1*a1 + w2*a2 + ... + w2n*a2n',
            x_limit = [min(x)-0.2, max(x)+0.2],
            y_limit = [min(y)-0.2, max(y)+0.2]) 

    for n in np.linspace(10, 300, 5, endpoint=False):
        n_random_pair_hidden_neurons_out_put(n, fitting_functions)

    plt.show()




def test_3():
    """二维情况，有3层网络，输出层一个神经元，隐藏层n对神经元，每对两个神经元组成，即0-1被分成了1/n个区间,
    每个区间可以是凸起和凹下表示，则n越大则输出越近似一条连续的曲线。输出层为一个神经元。
    1.y在-1到1之间，x在0-1之间，中间分成若干区间，随机拟合函数 f(x) = 0.2 + 0.4*x*x + 0.3*x*np.sin(15*x) + 0.05*np.cos(50*x)
    2.注意，最后是用输出层的输出拟合的f(x)，是输出层最终的输出
    3.要让最终输出层的结果拟合f(x)，输出层的输入应该是sigmod的反函数r-sigmod()和f(x)的符合，即r-sigmod(f(x))
    """
    # sigmod 反函数
    def reverse_sigmod(x):
        return np.log(x/(1.0-x))

    # 拟合的函数
    def fitting_functions(x):
        return reverse_sigmod(0.2 + 0.4*x*x + 0.3*x*np.sin(15*x) + 0.05*np.cos(50*x))

    def n_last_layer_out_put(n, fun):
        """画出最后一层的输出，fun是r-sigmod(f(x))，最终输出将是f(x)
        """
        y = None
        x = None

        for i in np.linspace(0, 1, n, endpoint=False):
            s1 = i; s2 = i+1.0/n; 
            h = fun(i) # 区间高度
            x, y1 = pair_hidden_neurons_out_put(s1, s2, h)

            if y == None:
                y = y1
            else:
                y += y1

        y = network2.sigmoid(y)

        plot_figure.plot_base(
            y_coordinate = [y],
            x_coordinate= [x],
            title='n = %d' % (n),
            x_lable='X',
            y_lable='Sigmod (z)',
            x_limit = [min(x)-0.2, max(x)+0.2],
            y_limit = [min(y)-0.2, max(y)+0.2]) 

    for n in np.linspace(10, 300, 5, endpoint=False):
        n_last_layer_out_put(n, fitting_functions)

    plt.show()

# test_1()
test_2()
# test_3()


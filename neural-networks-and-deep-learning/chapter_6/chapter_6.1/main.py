#!/usr/bin/python
# -*- coding: UTF8 -*-

import time
import sys
sys.path.append("../../common")

# import mnist_loader
import network3
# import command_line
# import plot_figure
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU

# 扩展数据集
import expand_mnist

# 测试函数
def test_0():
    """全连接 + 全连接 + softmax, 测试准确率97:80%
    """
    name = sys._getframe().f_code.co_name
    print name + "\n"

    training_data, validation_data, test_data = network3.load_data_shared()

    mini_batch_size = 10
    net = Network([
                FullyConnectedLayer(n_in=784, n_out=100),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1,
                validation_data, test_data)

def test_1():
    """全连接 + 卷积层 + 最大值混合层 + softmax，测试准确率98.48%
    """
    name = sys._getframe().f_code.co_name
    print name + "\n"

    training_data, validation_data, test_data = network3.load_data_shared()

    mini_batch_size = 10
    net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                             filter_shape=(20, 1, 5, 5),
                             poolsize=(2, 2)),
                SoftmaxLayer(n_in=20*12*12, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1,
                validation_data, test_data)

def test_2():
    """全连接 + 卷积层 + 最大值回合层 + 全连接 + softmax，测试准确率98:78% 
    """

    name = sys._getframe().f_code.co_name
    print name + "\n"

    training_data, validation_data, test_data = network3.load_data_shared()

    mini_batch_size = 10
    net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                             filter_shape=(20, 1, 5, 5),
                             poolsize=(2, 2)),
                FullyConnectedLayer(n_in=20*12*12, n_out=100),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1,
                validation_data, test_data)

def test_3():
    """全连接 + 卷积层 + 最大值回合层 + 卷积层 + 最大值混合层 + 全连接 + softmax，测试准确率99.09%
    """
    name = sys._getframe().f_code.co_name
    print name + "\n"

    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10

    net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                              filter_shape=(20, 1, 5, 5),
                              poolsize=(2, 2)),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                              filter_shape=(40, 20, 5, 5),
                              poolsize=(2, 2)),
                              FullyConnectedLayer(n_in=40*4*4, n_out=100),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1,
                validation_data, test_data)

def test_4():
    """全连接 + 卷积层 + 最大值回合层 + 卷积层 + 最大值混合层 + 全连接 + softmax
       激活函数:修正线性单元
       代价函数：L2规范化
       测试准确率：99.18%
    """

    name = sys._getframe().f_code.co_name
    print name + "\n"

    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10

    net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                                filter_shape=(20, 1, 5, 5),
                                poolsize=(2, 2),
                                activation_fn=ReLU),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                                filter_shape=(40, 20, 5, 5),
                                poolsize=(2, 2),
                                activation_fn=ReLU),
                FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
                
    net.SGD(training_data, 60, mini_batch_size, 0.03,
            validation_data, test_data, lmbda=0.1)

def test_5():
    """全连接 + 卷积层 + 最大值回合层 + 卷积层 + 最大值混合层 + 全连接 + softmax
       激活函数:修正线性单元
       代价函数：L2规范化
       训练数据：使用扩展数据集
       测试准确率：99.44%
    """
    name = sys._getframe().f_code.co_name
    print name + "\n"

    # 扩展数据集
    expand_mnist.expand_mnist_data()

    training_data, validation_data, test_data = \
        network3.load_data_shared("../../minst-data/data/mnist_expanded.pkl.gz")
    mini_batch_size = 10

    net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                                filter_shape=(20, 1, 5, 5),
                                poolsize=(2, 2),
                                activation_fn=ReLU),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                                filter_shape=(40, 20, 5, 5),
                                poolsize=(2, 2),
                                activation_fn=ReLU),
                FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
                
    net.SGD(training_data, 60, mini_batch_size, 0.03,
            validation_data, test_data, lmbda=0.1)

test_5()



#!/usr/bin/python
# -*- coding: UTF8 -*-

import time
import sys
sys.path.append("../../common")

# import mnist_loader
import network3
# import command_line
# import plot_figure


# 测试函数
def test_0():
    """全连接 + 全连接 + softmax, 测试准确率97:80%
    """

    import network3
    from network3 import Network
    from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

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

    import network3
    from network3 import Network
    from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

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

    import network3
    from network3 import Network
    from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

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
    """全连接 + 卷积层 + 最大值回合层 + 卷积层 + 最大值混合层 + 全连接 + softmax，测试准确率99:09%
    """

    import network3
    from network3 import Network
    from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

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

test_3()








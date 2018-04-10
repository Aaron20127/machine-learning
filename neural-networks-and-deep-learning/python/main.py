#!/usr/bin/python
# -*- coding: UTF8 -*-

import mnist_loader

# 读取50000训练数据，10000验证数据，10000测试数据
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

# training_data = training_data[:10]
# validation_data = validation_data[:10]
# test_data = test_data[:10]    

# print len(training_data), len(validation_data), len(test_data)

import network

# 设置神经网络层数和每层的节点数，这里首层和尾层的节点数只能是784和10
net = network.Network([784, 30, 10])

# 训练并测试
net.SGD(training_data, 30, 10, 3.0, test_data = test_data)

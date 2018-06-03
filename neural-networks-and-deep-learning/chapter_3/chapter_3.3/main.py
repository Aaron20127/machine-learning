#!/usr/bin/python
# -*- coding: UTF8 -*-

import time
import sys
sys.path.append("../../common")

import mnist_loader
import network2
import command_line
import plot_figure

# 读取50000训练数据，10000验证数据，10000测试数据
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 测试函数
def test_0():
        """使用1/sqrt(n)初始化权重的方法
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
        net.SGD(training_data, 30, 10, 0.1, lmbda=5.0,
                evaluation_data = validation_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_1():
        """使用标准正太分布初始化权重的方法
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data, 30, 10, 0.1, lmbda=5.0,
                evaluation_data = validation_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)


# cmd = ['-t', "0", "1", "-p", "ta", "tc", "ea", "ec" "-f", "test_0.net", "test_1.net"]
cmd = sys.argv[1:]

training_function = [test_0, test_1]

command_line.register_training_function(training_function)
command_line.execute_command(cmd)








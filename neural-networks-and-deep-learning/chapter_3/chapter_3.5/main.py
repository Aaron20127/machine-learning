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
        """eta = 0.025
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data, 30, 10, 0.025, lmbda=5.0,
                evaluation_data = validation_data,
                monitor_training_accuracy=False,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=False,
                monitor_evaluation_cost=False,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_1():
        """eta = 0.25
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data, 30, 10, 0.25, lmbda=5.0,
                evaluation_data = validation_data,
                monitor_training_accuracy=False,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=False,
                monitor_evaluation_cost=False,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_2():
        """eta = 2.5
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data, 30, 10, 2.5, lmbda=5.0,
                evaluation_data = validation_data,
                monitor_training_accuracy=False,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=False,
                monitor_evaluation_cost=False,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_3():
        """ momentum = False
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data, 100, 10, 0.05, lmbda=0,
                evaluation_data = validation_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_4():
        """momentum = True, 梯度下降加速
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost, momentum=True)
        net.SGD(training_data, 100, 10, 0.05, lmbda=0,
                evaluation_data = validation_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)


cmd = sys.argv[1:]
training_function = [test_0, test_1, test_2, test_3, test_4]
command_line.register_training_function(training_function)
command_line.execute_command(cmd)







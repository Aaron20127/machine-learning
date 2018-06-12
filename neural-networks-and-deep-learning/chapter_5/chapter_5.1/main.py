#!/usr/bin/python
# -*- coding: UTF8 -*-

import time
import sys
sys.path.append("../../common")

import mnist_loader
import network2
import command_line
import plot_figure

def test_0(training_data, validation_data, test_data):
        """ 一个隐藏层，对比不同深度之间的效果
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 500, 1000, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_1(training_data, validation_data, test_data):
        """两个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 500, 1000, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_2(training_data, validation_data, test_data):
        """3个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 500, 1000, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)


def test_3(training_data, validation_data, test_data):
        """4个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 500, 1000, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)


# cmd = ['-t', "0"] 
# cmd = ['-p', 'bg1', 'bgo', '-f', 'test_0.net', 'test_1.net', '-e']
# cmd = ['-p', 'bg1', 'bgo', '-f', 'test_0.net', 'test_1.net', '-a']

cmd = sys.argv[1:]
training_function = [test_0, test_1, test_2, test_3]

command_line.register_training_function(training_function)
command_line.execute_command(cmd)







#!/usr/bin/python
# -*- coding: UTF8 -*-

import json
import plot_figure
import sys

import base_module as bml
import mnist_loader

training_function = []

def register_training_function(function):
    global training_function
    training_function = function

## 训练命令
def train(arg):
    # 读取50000训练数据，10000验证数据，10000测试数据
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    for i in range(len(arg["function"])):
        training_function[int(arg['function'][i])](training_data, validation_data, test_data)

## 绘图命令
def plot(arg):
    """将缩写参数扩展成真实参数
    """
    type_dict = [
        ['ta', 'training accuracy', 'accuracy'],
        ['tc', 'training cost', 'cost'],   
        ['ea', 'evaluation accuracy', 'accuracy'],   
        ['ec', 'evaluation cost', 'cost'],
        ['bg1', 'hidden layer 1 bias gradient', 'bias gradient'],   
        ['bg2', 'hidden layer 2 bias gradient', 'bias gradient'], 
        ['bg3', 'hidden layer 3 bias gradient', 'bias gradient'], 
        ['bg4', 'hidden layer 4 bias gradient', 'bias gradient'],  
        ['bgo', 'output layer bias gradient', 'bias gradient']
    ]

    feature_arg_list = []
    file_list = arg["file"]
    arg_list = []
    feature_list = []
    arg_type_list = []

    # 从文件中读取数据
    for file in file_list:
        feature = bml.read_list_from_file(file)
        feature["title"]["file"] = file
        feature_list.append(feature)

    # 从参数中获取名称和类型
    for ag in arg["arg"]:
        valide = False
        for type in type_dict:      
            if ag == type[0]:
                arg_list.append(type[1])
                arg_type_list.append(type[2])
                valide = True
        if not valide:
            print ("Error: Invalid parameter \'%s\' !" % (ag))
            sys.exit()

    # 判断参数的类型，如果参数不是同一个类型则报错
    first_arg_type = arg_type_list[0]
    for arg_type in arg_type_list[1:]:
        if first_arg_type != arg_type:
            print ("Error: Parameter types are inconsistent !")
            sys.exit()

    # 将画每幅图需要的文件和参数放入在一起
    if arg["tag"] == 'each':
        for feature in feature_list:
            feature_arg = [[feature], arg_list]
            feature_arg_list.append(feature_arg)
    elif arg["tag"] == 'all':
        for ag in arg_list:
            feature_arg = [feature_list, [ag]] 
            feature_arg_list.append(feature_arg)
    else:
        print ("Invade command:", arg)
        sys.exit()

    plot_figure.plot_figure_from_feature_arg_list(feature_arg_list)

## 解析键值对
def getopt_from_argv(argv):
    opts = []
    while argv and argv[0].startswith('-') and argv[0] != '-':
        opt = ["",[]]
        opt[0] = argv[0] 
        argv = argv[1:]

        j = 0
        for i in range(len(argv)):
            if argv and (not argv[i].startswith('-')):
                opt[1].append(argv[i])
                j += 1
            else: break

        opts.append(opt)
        argv = argv[j:]
    return opts

## 外部调用解析命令
def execute_command(argv):
    opts = getopt_from_argv(argv)

    cmd_train = {
        "function" : []
    }

    cmd_plot = {
        "file" : [],
        "arg" : [],
        "tag"  : 'each'
    }

    for opt, arg in opts:

        if opt == '-h':
            # 
            print ('-p  : [ta ea] | [tc ec] | [bg1 bg2 ... bgo]， 需要画出的数据类型， 必须是同种数据类型 \n'+ \
                  '-t  : 0, 1, 2 ...， 执行第几个示例\n'+\
                  '-f  : *.net， 绘图时指定的文件\n' + \
                  '-e  : 同一个文件的同类型参数画在一幅图中\n' + \
                  '-a  : 不同文件的同类型参数画在一幅图中')

            # 执行第1个注册的训练函数
            print ('main.py -t 1' )
            # 将test_0.net和test_1.net的training_accuracy画在一起，
            # 将test_0.net和test_1.net的evaluation_cost画在一起
            print ('main.py -p ta ea -f test_0.net test1.net -a')
            # 将test_0.net的training_accuracy和evaluation_cost分别画在一起
            # 将test_1.net的training_accuracy和evaluation_cost分别画在一起
            print ('main.py -p ta ea -f test_0.net test1.net -e')
            sys.exit()
        elif opt in ("-t", "--train"):
            cmd_train["function"] = arg
        elif opt in ("-p", "--plot"):
            cmd_plot["arg"] = arg
        elif opt in ("-f", "--file"):
            cmd_plot["file"] = arg
        elif opt in ("-a", "--all"):
            cmd_plot["tag"] = 'all'
        elif opt in ("-e", "--each"):
            cmd_plot["tag"] = 'each'
        else:
            print ("Invade command:", opt)
            print ("Exit !")
            sys.exit()

    if cmd_train["function"]:
        train(cmd_train)

    if cmd_plot["arg"] and cmd_plot["file"]:
        plot(cmd_plot)

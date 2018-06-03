#!/usr/bin/python
# -*- coding: UTF8 -*-

import plot_figure

training_function = []

def register_training_function(function):
    global training_function
    training_function = function

## 训练命令
def train(arg):
    for i in range(len(arg["function"])):
        training_function[int(arg['function'][i])]()

## 绘图命令
def plot(arg):
    """将缩写参数扩展成真实参数
    """
    type_dict = {
        'ta': 'training accuracy',
        'tc': 'training cost',
        'ea': 'evaluation accuracy',
        'ec': 'evaluation cost'
    }
    file_list = arg["file"]
    type_list = [ type_dict[i] for i in arg["type"] ]
    plot_figure.plot_figure_from_file(file_list, type_list)

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
        "type" : []
    }

    for opt, arg in opts:

        if opt == '-h':
            # 执行第1个注册的训练函数
            print 'main.py -t 1' 
            # 将test_0.net和test_1.net的training_accuracy和evaluation_cost分别画到一幅图上
            print 'main.py -p ta ec -f test_0.net test1.net'
            # 训练之后直接将多个数据画到一幅图中
            print 'main.py -t 0 1 -p ta ec -f test_0.net test1.net'
            sys.exit()
        elif opt in ("-t", "--train"):
            cmd_train["function"] = arg
        elif opt in ("-p", "--plot"):
            cmd_plot["type"] = arg
        elif opt in ("-f", "--file"):
            cmd_plot["file"] = arg
        else:
            print "Invade command:", opt
            print "Exit !"
            sys.exit()

    if cmd_train["function"]:
        train(cmd_train)

    if cmd_plot["type"] and cmd_plot["file"]:
        plot(cmd_plot)

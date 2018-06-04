#!/usr/bin/python
# -*- coding: UTF8 -*-

"""只从.net文件绘制图型时使用plot.py更好，使用main.py会调用mnist_loader文件函数，使执行指令更慢
"""

import sys
sys.path.append("../../common")

import command_line

# cmd = ['-p', 'ta', 'tc', 'ea', 'ec', '-f', 'test_0.net']
cmd = sys.argv[1:]
command_line.execute_command(cmd)







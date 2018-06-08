#!/usr/bin/python
# -*- coding: UTF8 -*-

import time
import threading
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import network2

def h_print(string):
    print '\n--------------------------------------------'
    print string
    print '--------------------------------------------'

def c_print(count):
    print count
    print '--------------------------------------------'


#### matrix 矩阵的一些测试
class matrixTest(object):

    def test_1(self):
        """ 3维矩阵转换成二维矩阵
        """
        h_print('3维矩阵转换成二维矩阵')
        b = np.array([[[1],[2],[3]],[[4],[5],[6]]])

        c_print (b)
        c_print (b.reshape(2,3))
        c_print (b.reshape(2,3).transpose())

    def test_2(self):
        """ 使矩阵矩阵的列向量相加成一列，n*m乘上m*1全为1的矩阵，需要列相加就消除掉列
            使矩阵矩阵的列向量相加成一行，1*n全为1的矩阵乘上n*m，需要行相加就消除掉行
        """
        h_print('使矩阵矩阵的列向量相加成1列，或行向量加成1行')
        a = np.array([[1,1],[2,2],[3,3]])
        b = np.ones((a.shape[1],1))
        c = np.ones((1,a.shape[0]))

        c_print (a)
        c_print (b)
        c_print (c)
        c_print (np.dot(a,b)) # 列相加
        c_print (np.dot(c,a)) # 行相加

    def test_3(self):
        """ 生成一个任意维数矩阵，元素是等差数列或者是相同的值
        """
        h_print('生成一个任意维数矩阵，元素是等差数列')
        a = np.linspace(-1, 1, 10, endpoint=False)
        b = np.linspace(-1, 1, 10, endpoint=False).reshape(2,5) # 重新生成2*5的矩阵
        c = np.ones((2,3)) * 0.5 # 生成 2*3的元素为0.5的矩阵

        c_print (a)
        c_print (b)
        c_print (c)

    def test_4(self):
        print self
        print self.__class__

#### 画图测试 https://blog.csdn.net/qq_31192383/article/details/53977822
class plotTest:

    def plot_multiple_curve(self, y_coordinate, x_coordinate = [], line_lable = [], 
             line_color = [], title = '', x_lable = '', y_lable = ''):
        """
        描述：画一幅坐标曲线图，可以同时有多条曲线
        参数：y_coordinate （y坐标值，二元列表，例如[[1,2,3],[4,5,6]]，表示有两条曲线，每条曲线的y坐标为[1,2,3]和[4,5,6]）
             x_coordinate  (x坐标值，同y坐标值，如果不提供x坐标值，则默认是从0开始增加的整数)
             line_lable   （每条曲线代表的意义，就是曲线的名称，没有定义则使用默认的）
             line_color    (曲线的颜色，一维列表，如果比曲线的条数少，则循环使用给定的颜色；不给定时，使用默认颜色；
                            更多色彩查看 http://www.114la.com/other/rgb.htm)
             title        （整个图片的名称）
             x_lable      （x轴的含义）
             y_lable       (y轴的含义)
        """
        if (len(x_coordinate) > 0) and \
           (len(y_coordinate) != len(x_coordinate)):
            print "error：x坐标和y坐标不匹配！"
            return
        
        if (len(line_lable) > 0) and \
           (len(y_coordinate) != len(line_lable)):
            print "error：线条数和线条名称数不匹配，线条数%d，线条名称数%d！" % \
                  (len(y_coordinate),len(line_lable))     
            return

        if 0 == len(line_color):
            line_color = ['#9932CC', '#FFA933', '#FF4040', '#CDCD00',
                          '#CD8500', '#C0FF3E', '#B8860B', '#AB82FF']
            # print "info: 未指定色彩，使用默认颜色！"

        if len(y_coordinate) > len(line_color):
            print "warning: 指定颜色种类少于线条数，线条%d种，颜色%d种！" % \
                  (len(y_coordinate),len(line_color))

        # 如果没有给x的坐标，设置从0开始计数的整数坐标
        if 0 == len(x_coordinate):
            x_coordinate = [range(len(y)) for y in y_coordinate]

        # 如果没有给线条名称，则使用默认线条名称
        if 0 == len(line_lable):
            line_lable = ["line " + str(i) for i in range(len(y_coordinate))]

        plt.figure(figsize=(70, 35)) 
        ax = plt.subplot(111)

        for i in range(len(y_coordinate)):
            ax.plot(x_coordinate[i], y_coordinate[i], color = line_color[i%len(line_color)], \
                    linewidth = 2.0, label = line_lable[i])    

        ax.set_title(title, fontsize=14) # 标题
        ax.set_xlabel(x_lable, fontsize=14) # x坐标的意义
        ax.set_ylabel(y_lable, fontsize=14) # y坐标的意义
        # ax.set_xlim(self.get_min_and_max_in_list(x_coordinate)) # x坐标显示的宽度
        # ax.set_ylim(self.get_min_and_max_in_list(y_coordinate)) # y坐标的宽度
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="best", fontsize=14) # 线条的名称显示在右下角
        plt.grid(True) # 网格
    

    def test(self):
        self.plot_multiple_curve([[1,2,3],[6,5,6]],
                  line_lable = ['feature 1', 'feature 2'],
                  line_color = ['#9932CC', '#FFA933'],
                  title = 'Classification',
                  x_lable = 'Epoch',
                  y_lable = 'accuracy') 
        
        self.plot_multiple_curve([[2,2,3],[6,5,4]],
                line_lable = ['feature 1', 'feature 2'],
                line_color = ['#9932CC', '#FFA933'],
                title = 'Classification',
                x_lable = 'Epoch',
                y_lable = 'accuracy') 

        plt.show()


#### 线程测试
class threadTest:

    def action(self, arg):
        time.sleep(1)
        print arg

    def test(self):
        for i in xrange(10):
            arg = {"num" : i, "go" : "haha"}
            t =threading.Thread(target=self.action,args=(arg,))
            t.start()

        print 'main thread end!'

#### 使用静态变量
class staticVariableTest:

    def func(self):
        """a[0]是静态变量，只有把数据保存在list中才可以保留变量的值
        """
        a = [0]
        def funcn():
            a[0] += 1
            return a
        return funcn

    def test(self):
        f = self.func()
        print f()[0]
        print f()[0]
        print f()[0]

# threadTest().test()
# plotTest().test() 
# staticVariableTest().test()


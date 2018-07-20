#!/usr/bin/python
# -*- coding: UTF8 -*-

import time
import threading
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import network2

import math
import cPickle  
import gzip
import os.path
import os
import random

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

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

    def test_5(self):
        """矩阵做平移操作
        """
        c = np.arange(9)
        c = np.reshape(c, (3, 3))
        print(c)
        print '\n'
        print np.roll(c, shift=1, axis=0) #矩阵下移一步
        print '\n'
        print np.roll(c, shift=-1, axis=0) #矩阵上移一步
        print '\n'
        print np.roll(c, shift=1, axis=1) #矩阵右移一步
        print '\n'
        print np.roll(c, shift=-1, axis=1) #矩阵左移一步

    def test_6(self):
        """matrix和array的一些区别
           参考连接：https://blog.csdn.net/vincentlipan/article/details/20717163
        """
        ### 1. mat可以有I,T,H属性，array只有转置属性
        a = np.mat('1 0 0; 0 5 1; 0 0 1')
        print '原矩阵:\n', a
        print '转置：\n', a.T
        print '逆矩阵:\n', a.I
        print '基本数组:\n', a.A
        print '原矩阵乘上逆矩阵：\n',a*a.I

        # 共轭矩阵，存在虚数
        a = np.mat([[1, 0+1j],[0, 0]])
        print '\n原矩阵:\n', a
        print '共轭矩阵：\n', a.H

        # array只有转置属性
        b = np.array([[1,1],[0,2]])
        print '\n原矩阵:\n', b
        print '转置：\n', b.T

        c = np.mat(b) # 可以将np.array转换成matrix格式
        print '\n原矩阵：\n', c
        print '逆矩阵:\n', c.I  

        ### 2.矩阵乘法不一样
        a = np.array([[0, 1]])
        b = a.transpose()
        print '\narray中矩阵乘法:\n', np.dot(a, b)

        a = np.mat(a)
        b = np.mat(b)
        print 'matrix中中矩阵乘法:\n', a*b

        ### 3.同型矩阵对应元素相乘，好像只有array有这功能
        a = np.array([[0, 2]])
        print '\narray中同型矩阵对应元素相乘:\n', a*a

        # a = np.mat(a)
        # print 'matrix中同型矩阵对应元素相乘:\n', a.*a

        ### 4.**2含义不同，array表示矩阵元素平方，matrix表示两个相同的矩阵相乘
        a = np.array([[1, 2], [1, 2]])
        print '\n原矩阵:\n', a
        print 'array a**2:\n', a**2

        a = np.mat(a)
        print 'matrix a**2:\n', a**2

#### 列表和数组的赋值，浅拷贝和深拷贝
class copyTest:
    """要复制数组和列表，则直接使用copy.deepcopy()达到深复制的效果，
       copy.copy()只对数组有深复制的效果，列表是浅复制。
    """

    def test_1(self):
        """对于列表直接赋值相当于引用
        """
        print "1.引用，未开辟新的空间，赋值后b受影响"
        a = [1, 2, 3]
        b = a
        print 'b: ', b
        a[0] = 4
        print 'b: ', b

    def test_2(self):
        """列表有浅复制和深赋值方式，浅复制有两种方式，深复制使用deepcopy
        """
        print "\n2.浅复制只为列表的第一层开辟空间，若列表中还有列表则不会开辟空间，有两种浅复制方式"
        # 1.浅复制使用a[:]
        print "浅复制使用a[:]"
        a1 = [1, [1]]
        print 'a1: ', a1

        b1 = a1[:]
        a1[0] = 2
        a1[1].append(2)

        print 'a1: ', a1
        print 'b1: ', b1

        # 2.浅复制调用copy
        print "\n浅复制调用copy"
        import copy
        a2 = [1, [1]]
        print 'a2: ', a2

        b2 = copy.copy(a2)
        a2[0] = 2
        a2[1].append(2)
        print 'a2: ', a2
        print 'b2: ', b2

        # 3.深复制调用deepcopy
        print "\n深复制调用deepcopy"
        import copy
        a3 = [1, [1]]
        print 'a3: ', a3

        b3 = copy.deepcopy(a3)
        a3[0] = 2
        a3[1].append(2)
        print 'a3: ', a3
        print 'b3: ', b3

    def test_3(self):
        """数组和列表一样，有浅复制和深赋值方式，浅复制一种，深复制使用copy,deepcopy
        """
        print "\n3.数组和列表一样，有浅复制和深赋值方式，浅复制一种，深复制使用copy,deepcopy"
        # 1.浅复制使用a[:]
        print "浅复制使用a[:]，对三维以上无用"
        a1 = np.array([[[1]]]) # 三维数组
        print 'a1: ', a1

        b1 = a1[:]
        a1[0][0][0] = 2

        print 'a1: ', a1
        print 'b1: ', b1

        # 2.深复制调用copy
        print "\n深复制调用copy"
        import copy
        a2 = np.array([[[1]]]) 
        print 'a2: ', a2

        b2 = copy.copy(a2)
        a2[0][0][0] = 2
        print 'a2: ', a2
        print 'b2: ', b2

        # 3.深复制调用deepcopy
        print "\n深复制调用deepcopy"
        import copy
        a3 = np.array([[[1]]]) 
        print 'a3: ', a3

        b3 = copy.deepcopy(a3)
        a3[0][0][0] = 2
        print 'a3: ', a3
        print 'b3: ', b3

#### 画图测试 https://blog.csdn.net/qq_31192383/article/details/53977822
class plotTest:
    def plot_base(self, y_coordinate, x_coordinate = [], line_lable = [], 
                line_color = [], title = '', x_lable = '', y_lable = '',
                x_limit = [], y_limit = [], y_scale = 'linear', p_type = [],
                grad = False):
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
                x_limit       (x坐标的显示范围)
                y_scale       (y轴的单位比例，'linear'常规，'log'对数)
                p_type        (类型：line线条，scatter散点)
                grad          (网格)
        """

        if (x_coordinate and (len(y_coordinate) != len(x_coordinate))):
            print ("error：x坐标和y坐标不匹配！")
            sys.exit()
        
        if (line_lable and  (len(y_coordinate) != len(line_lable))):
            print ("error：线条数和线条名称数不匹配，线条数%d，线条名称数%d！" % \
                    (len(y_coordinate),len(line_lable)))     
            sys.exit()

        if not line_color:
            line_color = ['#9932CC', '#FF4040' , '#FFA933', '#CDCD00',
                            '#CD8500', '#C0FF3E', '#B8860B', '#AB82FF']
            # print "info: 未指定色彩，使用默认颜色！"

        if len(y_coordinate) > len(line_color):
            print ("warning: 指定颜色种类少于线条数，线条%d种，颜色%d种！" % \
                    (len(y_coordinate),len(line_color)))

        # plt.figure(figsize=(70, 35)) 
        plt.figure() 
        ax = plt.subplot(111)

        # 如果没有给x的坐标，设置从0开始计数的整数坐标
        if not x_coordinate:
            x_coordinate = [range(len(y)) for y in y_coordinate]

        # 如果没有给线条名称，则使用默认线条名称
        if not line_lable:
            line_lable = ["line " + str(i) for i in range(len(y_coordinate))]

        # 如果没有指定图形类型，默认画线条line
        if not p_type:
            p_type = ["line" for y in y_coordinate]

        for i in range(len(y_coordinate)):
            if p_type[i] == 'line':
                ax.plot(x_coordinate[i], y_coordinate[i], color = line_color[i%len(line_color)], \
                        linewidth = 2.0, label = line_lable[i])      
            elif p_type[i] == 'scatter': 
                ax.scatter(x_coordinate[i], y_coordinate[i],  s = 90, c=line_color[i%len(line_color)],\
                            linewidth = 2.0, alpha=0.6, marker='+', label = line_lable[i])
            else:
                print ("error：Invalid p_type %s！" % (p_type[i]))
                sys.exit()

        ax.set_title(title) # 标题
        ax.set_xlabel(x_lable) # x坐标的意义
        ax.set_ylabel(y_lable) # y坐标的意义
        ax.set_yscale(y_scale) # 'linear','log'
        ### 自适应轴的范围效果更好
        if x_limit: ax.set_xlim(x_limit) # x坐标显示的范围
        if y_limit: ax.set_ylim(y_limit) # y坐标显示范围
        
        # plt.xticks()
        # plt.yticks()
        plt.legend(loc="best") # 线条的名称显示在右下角
        if grad: plt.grid(True) # 网格

        # plt.savefig("file.png", dpi = 200)  #保存图片，默认png     
        # plt.show()

    def plot_base_3d(self, x_coordinate, y_coordinate, z_function, title = '',
                x_lable = '', y_lable = '', z_lable = '',
                x_limit = [], y_limit = [], z_limit = []):
        """绘制3D网格图
        """
        figure = plt.figure() 
        ax = Axes3D(figure)

        #网格化数据
        X, Y = np.meshgrid(x_coordinate, y_coordinate)
        Z = z_function(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(x_lable, fontsize=14) # x坐标的意义
        ax.set_ylabel(y_lable, fontsize=14) # y坐标的意义
        ax.set_zlabel(z_lable, fontsize=14) # z坐标的意义
        if x_limit: ax.set_xlim(x_limit) # x坐标显示的范围
        if y_limit: ax.set_ylim(y_limit) # y坐标显示的范围
        if z_limit: ax.set_zlim(z_limit) # z坐标显示的范围
    
    def picture_color_maps(self):
        """绘图的plt.imshow(img, cmap=None)的所有的cmap可选的值，即绘图的色域，
           可用在test_3中
        """
        # Have colormaps separated into categories:
        # http://matplotlib.org/examples/color/colormaps_reference.html
        cmaps = [('Perceptually Uniform Sequential', [
                    'viridis', 'plasma', 'inferno', 'magma']),
                ('Sequential', [
                    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
                ('Sequential (2)', [
                    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                    'hot', 'afmhot', 'gist_heat', 'copper']),
                ('Diverging', [
                    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
                ('Qualitative', [
                    'Pastel1', 'Pastel2', 'Paired', 'Accent',
                    'Dark2', 'Set1', 'Set2', 'Set3',
                    'tab10', 'tab20', 'tab20b', 'tab20c']),
                ('Miscellaneous', [
                    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
                    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

        nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        def plot_color_gradients(cmap_category, cmap_list, nrows):
            fig, axes = plt.subplots(nrows=nrows)
            fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
            axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

            for ax, name in zip(axes, cmap_list):
                ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
                pos = list(ax.get_position().bounds)
                x_text = pos[0] - 0.01
                y_text = pos[1] + pos[3]/2.
                fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

            # Turn off *all* ticks & spines, not just the ones with colormaps.
            for ax in axes:
                ax.set_axis_off()

        for cmap_category, cmap_list in cmaps:
            plot_color_gradients(cmap_category, cmap_list, nrows)

    def plot_picture(self, matrix, cmap=None, title=None, axis=True):
        """绘制矩阵图片
            matrix 是列表，每个元素代表一个图片的像素矩阵
            title  是列表，每个元素代表标题
            cmap   是色彩
        """
        def get_subplot_region_edge(num):
            for i in range(10000):
                if num <= i*i: 
                    return i

        total = len(matrix)
        edge = get_subplot_region_edge(total)
        plt.figure() 

        for i in range(total):
            ax = plt.subplot(edge, edge, i+1)  
            if title:
                ax.set_title(title[i], fontsize=14)

            if cmap:
                plt.imshow(matrix[i], cmap=cmap)
            else:
                plt.imshow(matrix[i])

            if not axis:
                plt.xticks([]) # 关闭图片刻度，必须放在imshow之后才生效
                plt.yticks([])

    def test(self):
        ### 1. 基本曲线图
        self.plot_base([[1,2,3],[6,5,6]],
                  line_lable = ['feature 1', 'feature 2'],
                  line_color = ['#9932CC', '#FFA933'],
                  title = 'Classification',
                  x_lable = 'Epoch',
                  y_lable = 'accuracy') 

        ### 2. 3D网格图
        def fun(x, y):
            return np.sin(3*x) + np.cos(3*y)

        self.plot_base_3d(np.arange(1, 10, 0.1), np.arange(1, 10, 0.1), fun,
                    x_lable='X', y_lable='Y', z_lable='Z')
        
        ### 3. 绘制图片的颜色域
        self.picture_color_maps()

        ### 4. 图片的读取和绘制
        """读取png图片，以不同色彩画出
           绘制随机灰度图
           绘制随机二值图
        """
        np.random.seed(19680801)
        img=mpimg.imread('src/stinkbug.png', format='png')
        # print img

        """ 1.因为原图是灰度图，即二维图，没有RGBA颜色分配，所以默认分配的颜色为viridis，
            每一个矩阵的值代表该颜色的深浅，所以给二维图分配不同的色彩可以得到不同颜色的效果图
            2.若原图是彩色图，即三位图，则自动生成原色彩的图片，不会失真
        """ 
        ### 绿色虫子(默认cmap = viridis)
        plt.figure()
        plt.imshow(img)

        ### 火红色虫子
        self.plot_picture([img], 'hot', title=['hot'])

        ### 灰色虫子
        self.plot_picture([img], 'gray', title=['gray'])

        ### 随机产生灰度图
        img = np.random.random((28, 28)) # 返回[0.0, 0.1)之间的随机数数组
        self.plot_picture([img], 'gray', title=['random gray']) 

        ### 二值图，灰色值只取0或1
        img = np.random.randint(low=0, high=2, size = (28,28)) # 随机生成0-1之间的整数
        self.plot_picture([img], 'binary', title=['random binary']) 

        # print img
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


#### 使用cPickle将图片保存成.plk文件，并将.plk文件作为图片输出
#### 学习用plt.imshow()绘制图片
class cPickleTest:

    def test_1(self):
        """将数据保成plk格式，并压缩成gz格式
        """
        a = np.array([[1,1],[2,2],[3,3]])
        b = {4:5,6:7}  

        #1. 不压缩直接保存成plk文件
        f = open('tmp/a.plk', "w")
        cPickle.dump((a, b), f, -1) # -1表示最优压缩
        f.close()  
        
        # 读取从plk文件获取数据
        f = open('tmp/a.plk', 'rb')
        c, d = cPickle.load(f)  
        f.close()  
        print c, d 

        #2. 压缩数据并保存成gz格式
        f = gzip.open('tmp/a.plk.gz', "w")
        cPickle.dump((a, b), f, -1)
        f.close()  
        
        #读取从plk.gz文件获取数据
        f = gzip.open('tmp/a.plk.gz', 'rb')
        c, d = cPickle.load(f)  
        f.close()  
        print c, d 


### 将mnistTest数据绘制成图片，并扩张数据集
class mnistTest:
    
    def plot_mnist(self, start, stop, figure_count, axis=False,
                   path='../minst-data/data/mnist.pkl.gz'):
        """描述：绘制mnsit数据成图片
           start: 从第几福图开始
           stop: 从第几福图结束
           figure_count：每幅图显示count个图片
           path: mnist路径
        """
        #读取从plk.gz文件获取数据
        f = gzip.open(path, 'rb')
        traning_data, _, _ = cPickle.load(f)  
        f.close()  

        total = stop - start + 1 # 画出第start到stop之间的图片
        matrix = []
        title = []

        if start > stop:
            print 'Error: start shouldn\'t begger than stop'
            sys.exit()      

        if stop >= len(traning_data[0]) or \
           total > len(traning_data[0]):
            print 'Error: the max length of traning_data is %d ' \
                    % (len(traning_data[0]))
            sys.exit()

        for i in range(total):
            matrix.append(traning_data[0][start + i].reshape((28,28))) # 转换成图片原来的矩阵
            title.append(traning_data[1][start + i])
            if (i+1) % figure_count == 0:
                plotTest().plot_picture(matrix, cmap='Greys', title = title, axis = axis)
                matrix = []
                title = []
        
        if matrix:
            plotTest().plot_picture(matrix, cmap='Greys', title=title, axis=axis)

        plt.show()

    def expand_mnist(self, start=None, stop=None,
                     src_path='../minst-data/data/mnist.pkl.gz',
                     dst_path='src/mnist_expanded.pkl.gz',
                     expand_count=4
                    ):
        """扩展数据集，将数据集上下左右平移，生成4份扩展数据，共生成5份数据
           start：从的第几个mnist数据开始扩展
           stop：结束至几个数据, 如果start和stop都为None,或者其中一个为none,则扩展所有数据
           src_path: 源数据位置
           dst_path: 保存数据位置
           expand_count: 单个数据扩展出多少份额外数据
        """
        f = gzip.open(src_path, 'rb')
        training_data, validation_data, test_data = cPickle.load(f)
        f.close()

        ## 如果start和stop都为None,或者其中一个为none,则扩展所有数据
        if start==None or stop==None:
            start = 0
            stop = len(training_data[0])-1

        ## 选择扩展数数据范围
        training_data = [training_data[0][start:(stop+1)],\
                         training_data[1][start:(stop+1)]]

        expanded_training_pairs = []

        ## 数据扩展的方法
        expanded_method_list = [
            (1,  0, 0), # 矩阵下移一行，将第一行置0
            (-1, 0, 27), # 矩阵上移，将最后一行置0
            (1,  1, 0), # 矩阵右移，将第一列置0
            (-1, 1, 27), # 矩阵左移，将最后一列置0

            (2,  0, 1), # 矩阵下移2行，将第一行和第二行置0
            (-2, 0, 26), # 矩阵上移2行，将最后两行置0
            (2,  1, 1), # 矩阵右移2列，将前两列置0
            (-2, 1, 26) # 矩阵左移2列，将最后两列置0
        ]

        if expand_count > len(expanded_method_list):
            print 'Error: the max number of expand_count is %d ' \
                   % (len(expanded_method_list))
            sys.exit()

        for x, y in zip(training_data[0], training_data[1]):
            expanded_training_pairs.append((x, y))
            image = np.reshape(x, (-1, 28))

            for d, axis, index in expanded_method_list[:expand_count]:
                #axis=0时，d>0下移,d<0上移。axis=1时，d>0右移，d<0左移
                new_img = np.roll(image, d, axis)
                if axis == 0: 
                    new_img[index, :] = np.zeros(28) # 将第n行换成0
                else: 
                    new_img[:, index] = np.zeros(28) # 将第n列换成0
                expanded_training_pairs.append((np.reshape(new_img, 784), y))
        # random.shuffle(expanded_training_pairs) # 随机打乱扩展数据
        expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
        f = gzip.open(dst_path, "w")
        cPickle.dump((expanded_training_data, validation_data, test_data), f)
        f.close()
        print("Saved expanded data, totle = %d " % (len(expanded_training_pairs)))

    def test(self):

        ###1.将mnist每个数据扩展expand_count份数据
        src_path='../minst-data/data/mnist.pkl.gz'
        dst_path='src/mnist_expanded.pkl.gz'

        # self.expand_mnist(0, 0, src_path, dst_path, expand_count=8) 
        self.plot_mnist(0, 8, 9, axis=True, path=dst_path) 

### 对png单个像素进行测试，主要是RBGA
class imagShowTest:
    """1.png格式图片的数组分二维和三维，三维格式为RGBA四个参数，最后一个参数是透明度
       2.二维格式是没有颜色的，默认灰度图，一个像素只有一个数字表示颜色的深浅。
         但是imshow可以指定二维图的颜色，且默认颜色是带绿色的viridis。若要绘制灰度图，
         则cmap=gray。
    """
    def test(self):
        plot_picture =  plotTest().plot_picture

        ### 1.浮点数格式
        # 单个像素RBGA，浮点数格式0-1，最后一位是透明度，不透明情况
        img = np.array([[[0.23529412, 0.36862746, 0.6156863 , 1.]]])
        plot_picture([img], cmap = 'gray', title = ['flaot32, a=1'])

        # 单个像素RBGA，浮点数格式0-1，最后一位是透明度，半透明情况
        img = np.array([[[0.23529412, 0.36862746, 0.6156863 , 0.5]]])
        plot_picture([img], cmap = 'gray', title = ['flaot32, a=0.5'])

        # 单个像素RBGA，浮点数格式0-1，最后一位是透明度，完全透明情况
        img = np.array([[[0.23529412, 0.36862746, 0.6156863 , 0.0]]])
        plot_picture([img], cmap = 'gray', title = ['flaot32, a=0.0'])

        ### 2.8bit格式
        # 单个像素RBGA，0-255，最后一位是透明度，不透明情况
        img = np.array([[[47, 79, 79 , 255]]])
        plot_picture([img], cmap = 'gray', title = ['uint8, a=255'])

        # 单个像素RBGA，0-255，最后一位是透明度，半透明情况
        img = np.array([[[47, 79, 79 , 128]]])
        plot_picture([img], cmap = 'gray', title = ['uint8, a=128'])

        # 单个像素RBGA，0-255，最后一位是透明度，完全透明情况
        img = np.array([[[47, 79, 79 , 0]]])
        plot_picture([img], cmap = 'gray', title = ['uint8, a=0.0'])

        ### 3.单个二维像素，imshow默认是viridis，可以指定成灰度gray
        # 1.一个像素只能是黑色
        img = np.array([[0.9]])
        plot_picture([img], cmap = 'gray', title = ['A pixel can only be black'])

        # 2.两个像素只能是黑白
        img = np.array([[0.4, 0.6]])
        plot_picture([img], cmap = 'gray', title = ['Two pixels can only be black and white'])

        # 2.3个像素以上才能有灰色
        img = np.array([[0.1, 0.5, 0.9]])
        plot_picture([img], cmap = 'gray', title = ['Gray can be seen above three pixels'])

        img = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        plot_picture([img], cmap = 'gray', title = ['Gray can be seen above three pixels'])

        plt.show()


### 空间坐标旋转，欧拉角和四元组的测试
### http://blog.miskcoo.com/2016/12/rotation-in-3d-space
### https://blog.csdn.net/lql0716/article/details/72597719

class eulerQuaternionsTest():

    def Rx(self, theta):
        """绕x轴旋转
        """
        cos = math.cos(theta)
        sin = math.sin(theta)
        return np.mat([[1,0,0],[0, cos, -sin],[0, sin, cos]])

    def Ry(self, theta):
        """绕y轴旋转
        """
        cos = math.cos(theta)
        sin = math.sin(theta)
        return np.mat([[cos,0,sin],[0, 1, 0],[-sin, 0, cos]])
    
    def Rz(self, theta):
        """绕z轴旋转
        """
        cos = math.cos(theta)
        sin = math.sin(theta)
        return np.mat([[cos,-sin,0],[sin, cos, 0],[0, 0, 1]])

    def rotate_around_any_axis(self, axis, theta, coordinate):
        """绕任意轴旋转
           axis：按照axis旋转的向量轴
           theta：延轴旋转的角度
           coordinate：需要被旋转的坐标
           返回：旋转后的坐标
        """

    def test(self):

        theta = math.pi / 2
        a1 = np.mat([[1],[0],[0]])

        # 每个轴旋转theta度的旋转矩阵
        Rx = self.Rx(theta)
        Ry = self.Ry(theta)
        Rz = self.Rz(theta)

        ### 1.单一变换，只按某个轴旋转
        ax = Rx*a1
        ay = Ry*a1
        az = Rz*a1

        print "1.分别绕xyz轴旋转做坐标变换"
        print "原坐标:\n", a1
        print "ax:\n",ax
        print "ay:\n",ay
        print "az:\n",az

        ### 2.旋转矩阵的性质，R(-theta) = R(theta).I = R(theta).T
        print "\n2.旋转矩阵的性质，以下矩阵值相同"
        print "R(-theta):\n",self.Rx(-theta)
        print "R(theta).I:\n",self.Rx(theta).I
        print "R(theta).T:\n",self.Rx(theta).T

        ### 3.不同的变换顺序导致结果不一样
        print "\n3.组合变换，不同的变换顺序导致结果不一样"
        print "原坐标:\n", a1
        print "\n绕zyx顺序变换:\n", Rx*Ry*Rz*a1
        print "\n绕zxy顺序变换:\n", Ry*Rx*Rz*a1
        print "\n绕xyz顺序变换:\n", Rz*Ry*Rx*a1

class polyfitTest:
    """多项式拟合函数，相当于机器学习中的多项式线性回归
       polyfit: 返回多项式系数数组，最前边的是高次项的系数
       poly1d:直接根据系数数组返回多项式函数
    """
    def test(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        y = np.array([2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0])

        #1. 3个系数，最高次数2次
        z = np.polyfit(x, y, 3)
        p3 = np.poly1d(z)

        #2. 30个系数，最高次数29次
        p30 = np.poly1d(np.polyfit(x, y, 30))

        xp = np.linspace(-2, 7, 100)
        _ = plt.plot(x, y, 'X', xp, p3(xp), '-', xp, p30(xp), '--')
        plt.ylim(-2,20)
        plt.show()


### 使用matplotlib进行动态画图测试
### 参考连接：https://blog.csdn.net/liang890319/article/details/52063941
class dynamicPlotTest:

    def interactiveTest(self):
        """可以在一个图上不停的画新的点，但是必须使用plt.pause对窗口进行刷新。
        刷新一次执行一次plt画图操作
        """
        plt.ion() #交互式界面
        for x in range(10):
            y = np.random.random()
            plt.scatter(x, y)
            plt.pause(1) # 1秒刷新一次画图操作
        
        while True:
            plt.pause(0.1) # 必须不停刷新窗口，不然窗口会消失

    def animationTest(self):
        """直接制作动画
        """
        pause = False

        def simData():
            t_max = 10.0
            dt = 0.05
            x = 0.0
            t = 0.0
            while t < t_max:
                if not pause:
                    x = np.sin(np.pi*t)
                    t = t + dt
                yield x, t
        
        def onClick(event):
            global pause
            pause ^= True
    
        def simPoints(simData):
            x, t = simData[0], simData[1]
            time_text.set_text(time_template%(t))
            line.set_data(t, x)
            return line, time_text
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line, = ax.plot([], [], 'bo', ms=10) # I'm still not clear on this stucture...
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 10)
        
        time_template = 'Time = %.1f s'    # prints running simulation time
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        fig.canvas.mpl_connect('button_press_event', onClick)
        ani = animation.FuncAnimation(fig, simPoints, simData, blit=False, interval=10,
            repeat=True)

        plt.show()

### 对运行时的系统路径进行测试
### https://blog.csdn.net/wangjianno2/article/details/48783127
class osPathTest:

    @staticmethod
    def test():
        #1.当前执行命令的目录
        print("os.getcwd():")
        cwd_path = os.getcwd()
        print(cwd_path)

        #2.获取绝对路径，如果给的目录是绝对路径形式，则返回不变，
        #  如果给的是一个相对目录，则系统会添加一个当前的工作目录到前边
        print("\nos.path.abspath():")
        print(os.path.abspath('src'))
        print(os.path.abspath('/src'))

        #3.将路径的最后一个目录和前边的分开
        print("\nos.path.split():")
        print(os.path.split(cwd_path))

        #4.得到去掉最末尾目录的路径
        print("\nos.path.dirname():")
        print(cwd_path)
        print(os.path.dirname(cwd_path))

        #5.得到末尾目录
        print("\nos.path.basename():")
        print(cwd_path)
        print(os.path.basename(cwd_path))

        #6.路径是否存在
        print("\nos.path.exists():")
        print(cwd_path)
        print(os.path.exists(cwd_path))     

        #7.是否是绝对路径
        print("\nos.path.isabs():")
        print(cwd_path)
        print(os.path.isabs(cwd_path))   

        #8.是否是一个文件
        print("\nos.path.isfile():")
        print(cwd_path)
        print(os.path.isfile(cwd_path))         

        #8.是否是一个目录   
        print("\nos.path.isdir():")
        print(cwd_path)
        print(os.path.isdir(cwd_path)) 

        #9.组合目录  
        print("\nos.path.join():")
        print(cwd_path)
        print(os.path.join(cwd_path, 'src')) 

        #10.添加模块路径
        print("\nsys.path.append, sys.path.insert:")
        sys.path.append("/test1")    #默认添加搜索路径到最后
        sys.path.insert(0, "/test2") #添加到指定的优先级，此处是表示添加到最前边
        sys.path.insert(2, "/test3") #添加到第三个位置
        print(sys.path)

if __name__=="__main__":
    # mnistTest().test()
    # cPickleTest().test_3()
    # threadTest().test()
    # staticVariableTest().test()
    # matrixTest().test_2()
    # plotTest().test()
    # imagShowTest().test()
    # eulerQuaternionsTest().test()
    # polyfitTest().test()
    # dynamicPlotTest().interactiveTest()
    # dynamicPlotTest().animationTest()

    osPathTest.test()




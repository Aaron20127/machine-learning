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

import cPickle  
import gzip

from mpl_toolkits.mplot3d import Axes3D

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

    def plot_base(self, y_coordinate, x_coordinate = [], line_lable = [], 
                line_color = [], title = '', x_lable = '', y_lable = '',
                x_limit = [], y_limit = []):
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

        if (len(x_coordinate) > 0) and (len(y_coordinate) != len(x_coordinate)):
            print "error：x坐标和y坐标不匹配！"
            return
        
        if (len(line_lable) > 0) and  (len(y_coordinate) != len(line_lable)):
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

        plt.figure(figsize=(70, 35)) 
        # ax = plt.subplot(221)
        ax = plt.subplot(111)

        # 如果没有给x的坐标，设置从0开始计数的整数坐标
        if 0 == len(x_coordinate):
            x_coordinate = [range(len(y)) for y in y_coordinate]

        # 如果没有给线条名称，则使用默认线条名称
        if 0 == len(line_lable):
            line_lable = ["line " + str(i) for i in range(len(y_coordinate))]

        for i in range(len(y_coordinate)):
            ax.plot(x_coordinate[i], y_coordinate[i], color = line_color[i%len(line_color)], \
                    linewidth = 2.0, label = line_lable[i])       

        ax.set_title(title, fontsize=14) # 标题
        ax.set_xlabel(x_lable, fontsize=14) # x坐标的意义
        ax.set_ylabel(y_lable, fontsize=14) # y坐标的意义
        ### 自适应轴的范围效果更好
        if x_limit: ax.set_xlim(x_limit) # x坐标显示的范围
        if y_limit: ax.set_ylim(y_limit) # y坐标显示范围
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="best", fontsize=14) # 线条的名称显示在右下角
        plt.grid(True) # 网格
        # plt.savefig("file.png", dpi = 200)  #保存图片，默认png     
        # plt.show()

    def plot_base_3d(self, x_coordinate, y_coordinate, z_function, title = '',
                x_lable = '', y_lable = '', z_lable = '',
                x_limit = [], y_limit = [], z_limit = []):
        """绘制3D网格图
        """
        figure = plt.figure(figsize=(24, 12)) 
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

    def plot_picture(self, matrix, cmap, title=None):
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
        plt.figure(figsize=(70, 35)) 

        for i in range(total):
            ax = plt.subplot(edge, edge, i+1)  
            if title:
                ax.set_title(title[i], fontsize=14)
            plt.imshow(matrix[i], cmap=cmap)
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

        ### 火红色虫子
        self.plot_picture([img], 'hot', title=['hot'])

        ### 灰色虫子
        self.plot_picture([1-img], 'Greys', title=['Greys'])

        ### 随机产生灰度图
        img = np.random.random((28, 28)) # 返回[0.0, 0.1)之间的随机数数组
        self.plot_picture([img], 'Greys', title=['random Greys']) 

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
    
    def plot_mnist(self, start, stop, figure_count, \
                    path = '../minst-data/data/mnist.pkl.gz'):
        """描述：绘制mnsit数据成图片
           start: 从第几福图开始
           stop: 从第几福图结束
           figure_count：总共由几个图显示
           path: mnist路径
        """
        #读取从plk.gz文件获取数据
        f = gzip.open(path, 'rb')
        traning_data, _, _ = cPickle.load(f)  
        f.close()  

        total = stop - start + 1 # 画出第start到stop之间的图片
        matrix = []
        title = []

        for i in range(total):
            matrix.append(traning_data[0][start + i].reshape((28,28))) # 转换成图片原来的矩阵
            title.append(traning_data[1][start + i])
            if (i+1) % figure_count == 0:
                plotTest().plot_picture(matrix, cmap='Greys', title = title)
                matrix = []
                title = []
        
        if matrix:
            plotTest().plot_picture(matrix, cmap='Greys', title = title)

        # plotTest().plot_picture([traning_data[0][0].reshape((28,28))], cmap='Greys', title = ['haha'])

        plt.show()


# cPickleTest().test_3()
# threadTest().test()
mnistTest().plot_mnist(0, 19, 16) 
# staticVariableTest().test()

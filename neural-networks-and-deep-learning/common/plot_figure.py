#!/usr/bin/python
# -*- coding: UTF8 -*-

#### Libraries
# Standard library
import json
import sys
sys.path.append("../common")

# Third-party libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import base_module as bml


def plot_base(y_coordinate, x_coordinate = [], line_lable = [], 
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

def plot_base_3d(x_coordinate, y_coordinate, z_function, title = '',
            x_lable = '', y_lable = '', z_lable = '',
            x_limit = [], y_limit = [], z_limit = []):

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

def extract_feature(feature_list, type):
    """从保存数据中提取信息
    """
    p_line_lable = []
    p_y_coordinate = []
    p_title = ''

    for feature in feature_list:
        p_title += json.dumps(feature["title"]) + '\n'
        for obj in feature["result"]:
            if obj["name"] == type :
                line_lable = ''
                if feature["title"].has_key("file"):
                    line_lable = feature["title"]["file"] + ", " + json.dumps(obj["min"]) + \
                                        ", " + json.dumps(obj["max"])
                else:
                    line_lable = json.dumps(obj["min"]) + \
                                        ", " + json.dumps(obj["max"])
                p_line_lable.append(line_lable)
                p_y_coordinate.append(obj["data"])
    return p_line_lable, p_y_coordinate, p_title    

def plot_figure_base(feature_list, type_list):
    "绘画基本图形"
    for type in type_list:
        p_y_lable =  type
        p_x_lable = "epoch"

        p_line_lable, p_y_coordinate, p_title = extract_feature(feature_list, type)

        if p_y_coordinate:
            plot_base(y_coordinate = p_y_coordinate,
                        line_lable = p_line_lable,
                        title = p_title,
                        x_lable = p_x_lable,
                        y_lable = p_y_lable)
        else:
            print "warning: no %s in file feature !" % type

def plot_figure_from_file(file_list, type_list):
    """将多福图像的同一特征绘制在一幅图像中
    """
    feature_list = []
    for file in file_list:
        feature = bml.read_list_from_file(file)
        feature["title"]["file"] = file
        feature_list.append(feature)

    plot_figure_base(feature_list, type_list)
    plt.show()

def plot_figure(feature_list, type_list):
    plot_figure_base(feature_list, type_list)
    plt.show()



#!/usr/bin/python
# -*- coding: UTF8 -*-


import json

def write_list_to_file(file, list):
    """将一个列表或字典转换成json字符号串存储到文件中"""
    obj_string = json.dumps(list)
    fo = open(file, "w")
    fo.write(obj_string)
    fo.close()
    return obj_string

def read_list_from_file(file):
    """将字符串转换成json对象，即列表或字典"""
    fo = open(file, "r")
    obj = json.loads(fo.read())
    fo.close()
    return obj


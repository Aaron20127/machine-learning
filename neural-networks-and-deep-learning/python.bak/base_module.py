#!/usr/bin/python
# -*- coding: UTF8 -*-


import json

def write_list_to_file(file, list):
    """将一个列表(list)转换成json字符号串存储到文件中"""
    json_stream = json.dumps(list)
    fo = open(file, "w")
    fo.write(json_stream)
    fo.close()
    return json_stream
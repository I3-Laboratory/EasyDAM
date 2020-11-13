# -*- coding: utf-8 -*-
# @File  : paser_xml.py
# @Author: SmartGx
# @Date  : 19-6-22 下午4:28
# @Desc  : 解析xml文件
import os
import numpy as np
from xml.etree import ElementTree as ET


def convert_annotation(file, writer):
    tree = ET.parse(open(file))
    root = tree.getroot()

    try:
        line = file.split(os.path.sep)[-1].split('.')[0]
    except:
        print('Can not get xml files!! Please check it')
        exit(0)

    # 解析文件中的每个目标
    boxes = ''
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text                 # 获得类别名称
        if cls != 'orange' or int(difficult)==1:
            continue
        cls_id = 0                                  # 类别id
        xmlbox = obj.find('bndbox')                 # 获得(x1,y1,x2,y2)坐标
        box = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        box = str(box).replace(' ', '') + ' '
        boxes += box
    line = '"{}":{}'.format(line, boxes)
    output = line.strip()
    writer.write(output+'\n')


if __name__ == '__main__':
    root = '../data/fruit_04/orange_set04_1/outputs'
    savePath = './orange_set04_1.txt'

    writer = open(savePath, 'a')
    files = os.listdir(root)
    # files.sort(key=lambda x:int(x[:-4]))
    files.sort()
    for file in files:
        print(file)
        filePath = os.path.join(root, file)
        convert_annotation(filePath, writer)
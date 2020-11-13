# -*- coding: utf-8 -*-
# @File  : modify_office_annotions.py
# @Author: SmartGx
# @Date  : 19-5-31 下午4:58
# @Desc  : 将label文件中Ground Truth坐标从(x,y,w,h)形式变为(x1,y1,x2,y2)
import os
import numpy as np

annotions_xywh = './annotions_xywh'

for file in os.listdir(annotions_xywh):
    oldPath = os.path.join(annotions_xywh, file)
    savePath = os.path.join('./annotions_xyxy', file)

    f = open(savePath, 'a')

    data = open(oldPath, 'r').readlines()
    for line in data:
        key, val = line.strip().split(':')
        # 坐标类型(topl_x, topl_y, w, h)
        val = np.array([eval(v) for v in val.strip().split(' ')])

        # 将坐标转化为(x1, y1, x2, y2)
        val[:, 2] = val[:, 0] + val[:, 2]
        val[:, 3] = val[:, 1] + val[:, 3]

        res = ''
        for each in val:
            v = str(tuple(each))
            res += (v+' ')
        text = res.strip().replace(', ', ',')
        new_line = "{}:{}\n".format(key, text)
        f.write(new_line)
    f.close()
    print('[INFO] Writing {} Finishing!!'.format(file))

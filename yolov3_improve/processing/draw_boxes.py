# -*- coding: utf-8 -*-
# @File  : draw_boxes.py
# @Author: SmartGx
# @Date  : 19-5-24 下午11:50
# @Desc  : 在原始图像上依据Ground Truth绘制边框，验证标签坐标的准确性
import cv2
import os
import numpy as np

rgbPath = './epfl_corridor_3/RGB'
depthPath = './epfl_corridor_3/Depth'
labels = './epfl_corridor_3/epfl_corridor_3.txt'

# baseDir = '31'

# rgbPath = '/home/guo/workspace/Dataset/RGB-D/New_UM/' + baseDir + '/RGB'
# depthPath = '/home/guo/workspace/Dataset/RGB-D/New_UM/' + baseDir + '/Depth'
# labels = '/home/guo/workspace/Dataset/RGB-D/New_UM/' + baseDir + '/'+ baseDir + '.txt'

rgb_files = sorted(os.listdir(rgbPath))
rgb_filePath = [os.path.join(rgbPath, each) for each in rgb_files]
d_files = sorted(os.listdir(depthPath))
d_filePath = [os.path.join(depthPath, each) for each in d_files]


for idx in range(len(rgb_files)):

	rgb = cv2.imread(rgb_filePath[idx])
	depth = cv2.imread(d_filePath[idx])

	label = open(labels, 'r').readlines()[idx]
	data = label.strip().split(':')[1].split(' ')
	if data == ['']:
		continue
	boxes = np.array([eval(item) for item in data])

	print(rgb_filePath[idx])
	print(d_filePath[idx])
	print(label)

	# label文件中坐标类型为(tl_x, tl_y, w, h)
	for p in boxes:
	    # x,y,w,h = p
	    # x1 = int(max(0, x-w/2))
	    # y1 = int(max(0, x-h/2))
	    # x2 = int(max(0, x+w/2))
	    # y2 = int(max(0, x+h/2))
	    # cv2.rectangle(rgb, (x,y), (x+w,y+h), (0, 255, 0), 2, cv2.LINE_AA)
	    # cv2.rectangle(depth, (x,y), (x+w,y+h), (0, 255, 0), 2, cv2.LINE_AA)

	    x1,y1,x2,y2 = p
	    x1 = int(max(0, x1))
	    y1 = int(max(0, y1))
	    x2 = int(max(0, x2))
	    y2 = int(max(0, y2))
	    cv2.rectangle(rgb, (x1,y1), (x2,y2), (0, 255, 0), 2, cv2.LINE_AA)
	    cv2.rectangle(depth, (x1,y1), (x2,y2), (0, 255, 0), 2, cv2.LINE_AA)


	cv2.imshow('RGB', rgb)
	cv2.imshow('Depth', depth)
	cv2.waitKey(0)
# -*- coding: utf-8 -*-
# @File  : concat_target_files.py
# @Author: SmartGx
# @Date  : 19-5-24 下午2:12
# @Desc  : 处理UM数据集的样本图像和标签文件
import os
import shutil

# 合并所有标注文件, 写入新文件
def modifyUmAnnotions(annotions_path, outputPath, write=False):
    files = os.listdir(annotions_path)
    hash_map = dict()
    for path in files:
        filePath = os.path.join(annotions_path, path)
        for line in open(filePath, 'r').readlines():
            data = line.strip().split(' ')
            org_key = data[0]
            first, end = org_key.split('.')
            # 取时间戳第二部分的前4位
            end = end[:4]
            key = first + '.' + end
            val = str(tuple([eval(x) for x in data[-4:]])).replace(' ', '')
            if key not in hash_map:
                hash_map[key] = val
            else:
                hash_map[key] += ' '
                hash_map[key] += val
    if write:
        if os.path.exists(outputPath):
            os.remove(outputPath)
        f = open(outputPath, 'a')
        for k, v in sorted(hash_map.items()):
            text = '"{}":{}\n'.format(k, v)
            f.write(text)
    return hash_map

# 提取label文件中的所有图像
def copyUmImages(fileDir, hash_map, copyImgPath, write=False):
    # 读取所有图片文件
    files = os.listdir(fileDir)
    # 判断复制文件夹是否存在
    if not os.path.exists(copyImgPath):
        os.mkdir(copyImgPath)

    # 循环所有文件进行查找
    counter = 0
    for file in sorted(files):
        srcFile = os.path.join(fileDir, file)
        if file.endswith('.png'):
            # 取文件名的前15位作为新文件名
            name = file[:15]
            dstFile = os.path.join(copyImgPath, name) + '.png'
            if name in hash_map.keys():
                if write:
                    counter += 1
                    shutil.copyfile(srcFile, dstFile)  # 复制文件到副本文件夹
                    print('Copy %s to %s ...' % (srcFile, dstFile))
    print('[INFO] Find {} name in labels, convert {} imgs'.format(len(hash_map), counter))

# 每八张图提取一张样本图像
def extractPer8Imgs(rgbPath, newPath):
    counter = 0
    if not os.path.exists(newPath):
        os.mkdir(newPath)
    for i, file in enumerate(sorted(os.listdir(rgbPath))):
        srcPath = os.path.join(rgbPath, file)
        dstPath = os.path.join(newPath, file)
        if i % 8 == 0:
            counter += 1
            shutil.copyfile(srcPath, dstPath)
    print('[INFO] Eatracted {} imgs'.format(counter))


if __name__ == '__main__':
    annotions_path = './images/annotions'
    outputPath = './labels.txt'
    # 原始RGB和Depth图像
    rgbPath = './images/rgb'
    depthPath = './images/depth'
    # 根据复制后的RGB图像
    copyRgbPath = './RGB'
    copyDepthPath = './Depth'

    # 待提取路径和提取图像后的新路径
    baseName = '_2011-06-20-18-59-56_0'
    origRgbPath = '/home/guo/workspace/Dataset/RGB-D/UM dataset/' + baseName + '/rgb'
    extractedRgbPath = '/home/guo/workspace/Dataset/RGB-D/UM dataset/' + baseName + '/exact_rgb'
    extractPer8Imgs(origRgbPath, extractedRgbPath)

    origDepthPath = '/home/guo/workspace/Dataset/RGB-D/UM dataset/' + baseName + '/depth'
    extractedDepthPath = '/home/guo/workspace/Dataset/RGB-D/UM dataset/' + baseName + '/exact_depth'
    extractPer8Imgs(origDepthPath, extractedDepthPath)


    # hash_map = modifyUmAnnotions(annotions_path, outputPath, write=False)
    # copyUmImages(rgbPath, hash_map, copyRgbPath, write=False)
    # copyUmImages(depthPath, hash_map, copyDepthPath, write=False)
# 文件功能：修改dark和cloth图像名称和注释文件
import os

# 修改dark和cloth数据集注释文件格式
def modify_dark_cloth_annotions(old_file, new_file):
    f = open(old_file, 'r')
    new_f = open(new_file, 'a')
    for line in f.readlines():
        swith, context = line.strip().split(':')
        fileName = '"' + swith + '.png' + '"'
        context = context[:-1]
        text = fileName+':'+context+'\n'
        new_f.write(text)


# 批量修改原始dark和cloth数据集图像名称
def change_dark_cloth_name(fileDir, pred='rgb_'):
    files = os.listdir(fileDir)
    # print(filnames)
    for file in files:
        if file.endswith('.png'):
            pre, swi = file.strip().split('_')
            src = os.path.join(fileDir, file)
            dst = os.path.join(fileDir, pred + swi)
            
            os.rename(src, dst)
            print('converting %s to %s ...' % (src, dst))


# 批量修改dark和cloth数据集图像名称
def change_repair_name(fileDir):
    files = os.listdir(fileDir)
    # print(filnames)
    for file in files:
        if file.endswith('.png'):
            idx = file.strip().split('_')[1]
            src = os.path.join(fileDir, file)
            dst = os.path.join(fileDir, idx.zfill(6) + '.png')
            
            os.rename(src, dst)
            print('converting %s to %s ...' % (src, dst))


def change_names(fileDir, post='.png'):
    files = os.listdir(fileDir)
    # print(filnames)
    for file in files:
        if file.endswith(post):
            pre = file.strip().split('.')[0]
            src = os.path.join(fileDir, file)
            dst = os.path.join(fileDir, pre.zfill(6)+post)
            
            os.rename(src, dst)
            print('converting %s to %s ...' % (src, dst))


if __name__ == '__main__':
    # modify_dark_cloth_annotions('./31.txt', './new_31.txt')

    # base = '2011-06-24-18-20-47_1'
    # data_root = '/home/guo/workspace/Dataset/RGB-D/RGB_HHA/' + base + '/Depth'

    # data_root = '/home/guo/workspace/Dataset/RGB-D/01_micc_crowd_counting/tools/label'
    # change_names(data_root, post='.dat')

    root = '/home/guo/workspace/Dataset/RGB-D/super_640_480/Queue'
    change_repair_name(root)
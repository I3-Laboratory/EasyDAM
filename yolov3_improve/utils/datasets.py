import glob
import math
import os
import random
import json

import cv2
import numpy as np
import torch

# from torch.utils.data import Dataset
from utils.utils import xyxy2xywh
from collections import OrderedDict


class LoadImages:  # for inference
    def __init__(self, path, img_size=416, read_txt=False):
        if os.path.isdir(path):
            self.files=[]
            # print('xxxxxxxxxxx' * 6,os.path.isdir(path))
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            #self.files = sorted(glob.glob('%s/*.*' % path))
            rgbList = [os.path.join(path, name) for name in os.listdir(path)]
            # print("(1)", rgbList)
            self.files = rgbList
            # for i in range(len(rgbList)):
            #     Path="/home/guo/workspace/chenkaizhen/yolov3-lab/" + path+str(i).zfill(5)+'.png'
            #     self.files.append(Path)


            # print('ssssssss'*10,os.listdir(path))

                # 如果需要忽略某些文件夹，使用以下代码
                # if s == "xxx":
                # continue

            #self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, newDir))
            #print('xxxxxxxxxxx'*6,self.files)
        elif os.path.isfile(path) and path.endswith('.txt'):
            self.files = []
            lines = open(path, 'r').readlines()
            #print('%%%%'*10,lines)
            for line in lines:
                rgb, d = line.strip().split('\t')
                self.files.append(rgb)
        elif os.path.isfile(path):
            self.files = [path]
        # print("(1)", rgbList)
        self.nF = len(self.files)  # number of image files
        self.height = img_size

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]
        #print('xxxxxxxxx'*10,img_path )
        # Read image
        img0 = cv2.imread(img_path)  # BGR
        # print("(1)", img0.shape)
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        # print("(2)", img.shape)

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, img_size=416):
        self.cam = cv2.VideoCapture(0)
        self.height = img_size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read image
        ret_val, img0 = self.cam.read()
        assert ret_val, 'Webcam Error'
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return 0


class LoadImagesAndLabels:  # for training
    def __init__(self, data_root, batch_size=1, img_size=608, multi_scale=False, augment=False, train=True, test=False):
        self.rgb_files = []
        # self.depth_files = []
        self.labels_files = []

        # label_path = os.path.join(data_root, 'json_label.json')  #fake apple datasets
        # label_path = os.path.join(data_root, 'pesudo_apple_test.json')
        label_path = os.path.join(data_root, 'labels.json')
        # label_path = os.path.join(data_root, 'test_model.json')    #real apple datasets
        # label_path = os.path.join(data_root, 'train_model.json')  # real apple datasets
        # label_path = os.path.join(data_root, 'pesudo_label.json')
        # label_path = os.path.join(data_root, 'MineApple.json')

        self.label_dict = json.loads(open(label_path, 'r').read())

        #加载数据集划分txt文件
        """=======real apple ============="""
        if train and not test:
            data_txt = os.path.join(data_root, 'fruit_train.txt')
        if not train and test:
            data_txt = os.path.join(data_root, 'fruit_test.txt')
            # data_txt = os.path.join(data_root, 'fruit_train.txt')
        if not train and not test:
            data_txt = os.path.join(data_root, 'eval.txt')
        """==============================="""

        """==============fake apple========"""
        # if train and not test:
        #     data_txt = os.path.join(data_root, 'all_fruit_datasets/fruit_train.txt')
        # if not train and test:
        #     data_txt = os.path.join(data_root, 'all_fruit_datasets/fruit_test.txt')
        # if not train and not test:
        #     data_txt = os.path.join(data_root, 'eval.txt')
        """==================================="""

        """==============更改读取图像路径方法=========="""
        # if pesudo_train:
        #     data_txt = os.path.join(data_root, "fruit_pesudo_train.txt")


        """========================================="""

        for line in open(data_txt).readlines():     #原配置是双流网络，则txt中每一行包括RGB图像路径和其对应的Depth图像路径
            # r, d = line.strip().split('\t')
            r = line.strip()
            self.rgb_files.append(r)
            # self.depth_files.append(d)      #如果是单流网络，注释这一行
        # print("(1)", self.rgb_files)
        # 其他参数
        self.nF = len(self.rgb_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment


    def __iter__(self):
        self.count = -1
        # np.random.permutation()：返回一个新的打乱数组,  np.random.shuffle()：在原有数组的基础上打乱
        self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)

        # 是否多尺度训练
        if self.multi_scale:
            # Multi-Scale YOLO Training
            height = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
        else:
            # Fixed-Scale YOLO Training
            height = self.height

        # 存储所有rgb，depth和对应label
        rgb_all = []
        # depth_all = []      #如果是单流网络，注释这一行
        labels_all = []
        # 提取一个batch的数据
        for index, files_index in enumerate(range(ia, ib)):
            rgb_path = self.rgb_files[self.shuffled_vector[files_index]]
            # print("(1)", rgb_path)
            # depth_path = self.depth_files[self.shuffled_vector[files_index]]        #如果是单流网络，注释这一行
            #print('tttttt' * 10, rgb_path)
            label = self.label_dict[rgb_path]
            # print('tttttt' * 10, rgb_path)
            # label = self.label_dict[depth_path]
            # print('RGB:{}\tDepth:{}'.format(rgb_path, depth_path))
            # print('tttttt' * 10, depth_path)
            """?????????????/"""
            """
            depth = cv2.imread(rgb_path)  # BGR3通道        #如果是单流网络，注释这一行
            rgb = cv2.imread(depth_path) # Depth3通道
            """
            """?????????????/"""
            # print("(1）：", rgb_path)
            rgb = cv2.imread(rgb_path)  # BGR3通道
            # print("(1）：", rgb)
            img0 = rgb
            # depth = cv2.imread(depth_path, 0) # Depth单通道
            # depth = np.expand_dims(depth, 2)
            # print('tttttt' * 10,depth)
            """========修改==========="""
            # if rgb is None or depth is None:
            if rgb is None:
                continue

            # 变换到HSV空间进行图像增强
            augment_hsv = True
            if self.augment and augment_hsv:
                # SV augmentation by 50%
                fraction = 0.50
                img_hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=rgb)

            # 将RGB和Depth图像resize到(416x416)
            h, w, _ = rgb.shape
            rgb, ratio, padw, padh = letterbox(rgb, height=height)
            # depth, _, _, _ = letterbox(depth, height=height)    #如果是单流网络，注释这一行

            # 调整labels
            if label != '':
                # print([item for item in label.strip().split(',')])
                # labels0 = np.array([eval(item) for item in label.strip().split(' ')]).reshape(-1,4)
                """============修改========="""
                # labels0 = np.array([eval(item) for item in label.strip().split(' ')]).reshape(-1,4)
                labels0 = np.array([eval(item.strip()) for item in label]).reshape(-1,4)
                """===========修改结束========="""
                # Normalized xywh to pixel xyxy format
                labels = np.zeros(shape=(labels0.shape[0], 5), dtype=np.float32)
                labels[:, 1] = ratio * labels0[:, 0] + padw
                labels[:, 2] = ratio * labels0[:, 1] + padh
                labels[:, 3] = ratio * labels0[:, 2] + padw
                labels[:, 4] = ratio * labels0[:, 3] + padh
            else:
                labels = np.array([])

            # 可视化测试Ground Truth的box正确性
            # for each in labels:
            #     cv2.rectangle(rgb, (each[1], each[2]), (each[3], each[4]), (0, 255, 0), 2)
            # cv2.imshow('Test', rgb)
            # cv2.waitKey(0)

            # 图像裁剪平移旋转
            if self.augment:
                rgb, labels, M = random_affine(rgb, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
                # depth, _, _ = random_affine(depth, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))     #如果是单流网络，注释这一行

            nL = len(labels)
            if nL > 0:
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy()) / height

            if self.augment:
                # random left-right flip
                lr_flip = True
                if lr_flip & (random.random() > 0.5):
                    rgb = np.fliplr(rgb)
                    # depth = np.fliplr(depth)    #如果是单流网络，注释这一行
                    if nL > 0:
                        labels[:, 1] = 1 - labels[:, 1]

                # random up-down flip
                ud_flip = False
                if ud_flip & (random.random() > 0.5):
                    rgb = np.flipud(rgb)
                    # depth = np.flipud(depth)    #如果是单流网络，注释这一行
                    if nL > 0:
                        labels[:, 2] = 1 - labels[:, 2]

            rgb_all.append(rgb)
            # depth_all.append(depth)     #如果是单流网络，注释这一行
            labels_all.append(torch.from_numpy(labels))
        # RGB-Normalize
        # print('tttttt'*10,rgb_all)
        rgb_all = np.stack(rgb_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
        rgb_all = np.ascontiguousarray(rgb_all, dtype=np.float32)
        rgb_all /= 255.0


        # Depth-3通道
        """如果是单流网络，注释这一段代码"""
        # depth_all = np.stack(depth_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
        # depth_all = np.ascontiguousarray(depth_all, dtype=np.float32)
        # depth_all /= 255.0

        # img_shape = [bs, rgb, 416, 416], labels_shape = [bs, [class_id,x,y,w,h]]
        # return torch.from_numpy(rgb_all),torch.from_numpy(depth_all), labels_all
        # return torch.from_numpy(rgb_all), labels_all
        return torch.from_numpy(rgb_all), labels_all, img0

    def __len__(self):
        return self.nB  # number of batches


def letterbox(img, height=416, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 1:5].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy, 0, height, out=xy)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return imw, targets, M
    else:
        return imw


def convert_tif2bmp(p='../xview/val_images_bmp'):
    import glob
    import cv2
    files = sorted(glob.glob('%s/*.tif' % p))
    for i, f in enumerate(files):
        print('%g/%g' % (i + 1, len(files)))
        cv2.imwrite(f.replace('.tif', '.bmp'), cv2.imread(f))
        os.system('rm -rf ' + f)

# if __name__ == '__main__':
    dataLoader = LoadImagesAndLabels('../data/rgbd', batch_size=1,augment=True, img_size=416)
    for i,(rgb, depth, labels) in enumerate(dataLoader):
        print(labels)
        break
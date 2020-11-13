import argparse
import shutil
import time
from pathlib import Path
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def detect(cfg,
           weights,
           images,
           output,
           img_size=416,
           conf_thres=0.5,
           nms_thres=0.4,
           save_txt=False,
           save_images=True,
           webcam=False):
    device = torch_utils.select_device()

    basepath = os.getcwd()

    if not os.path.exists(output):
        os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)
    #结束训练
    model.to(device).eval()

    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size, read_txt=opt.read_txt)

    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]
    label_dict = OrderedDict()
    
    for i, (path, img, im0) in enumerate(dataloader):
        print("(1)", path)
        t = time.time()
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return

        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshol

        """==========draw GT ============="""
        # TestImageName = path.strip().split("/")[-1]
        # TestImageName = os.path.join(BaseImagePath, TestImageName)
        # gt_box = label_dict[TestImageName]

        # for i in gt_box:
        #     i = re.findall(r'[(](.*?)[)]', i)
        #     xmin = int(i[0].strip().split(",")[0])
        #     ymin = int(i[0].strip().split(",")[1])
        #     xmax = int(i[0].strip().split(",")[2])
        #     ymax = int(i[0].strip().split(",")[3])
        #     cv2.rectangle(im0, (xmin, ymin), (xmax, ymax), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
        """================================"""

        if len(pred) > 0:
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

            if detections is None:
                if save_images:
                    cv2.imwrite(save_path, im0)
                elif webcam:
                    cv2.imshow(weights, im0)
                continue
            # Rescale boxes from 416 to true image size
            detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape)

            # Print results to screen
            unique_classes = detections[:, -1].cpu().unique()
            for c in unique_classes:
                n = (detections[:, -1].cpu() == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                # if int(cls) != 0:
                    # continue
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write('%g %g %g %g %g %g\n' %
                                   (x1, y1, x2, y2, cls, cls_conf * conf))
                label = ''
                cv2.rectangle(im0, (x1, y1), (x2, y2), (255,0,0), thickness=2, lineType=cv2.LINE_AA)
            """====制作伪标签文件==========="""
            detections = detections.cuda().data.cpu().numpy()
            label_list = []
            for i in range(len(detections)):
                xmin = detections[i][0]
                ymin = detections[i][1]
                xmax = detections[i][2]
                ymax = detections[i][3]
                loc = "("+str(int(xmin))+","+str(int(ymin))+","+str(int(xmax))+","+str(int(ymax))+")"
                label_list.append(loc)
            filepath = os.path.join(basepath, path)
            label_dict[filepath] = label_list
            """======制作结束==========="""

        dt = time.time() - t
        print('Spend time: {:.3f}s'.format(dt))

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

    """========obtain json file=============="""
    pesudo_json = json.dumps(label_dict)
    with open("data_pse/labels.json", "w") as j:
        j.write(pesudo_json)
        print('[INFO] Writing labels.json finishing!!')
    """======================================="""

    """=========json to txt======="""
    label_dict = json.loads(open("data_pse/pseudo_label.json", 'r').read())
    if os.path.exists("data_pse/fruit_train.txt"):
        os.remove("data_pse/fruit_train.txt")
    file_txt_write = open("data_pse/fruit_train.txt", "w")
    for key, value in label_dict.items():
        file_txt_write.writelines(key)
        file_txt_write.writelines("\n")
    """=========================================="""

    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-orange.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--output', type=str, default='data_pse/output', help='path to save path')
    parser.add_argument('--images', type=str, default='data_pse/unlabeled_trainset/', help='path to images')
    parser.add_argument('--webcam', type=bool, default=False, help='webcam to detections or not')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    parser.add_argument('--read_txt', type=str, default=False, help='read image from test.txt')
    opt = parser.parse_args()
    print(opt)
    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            opt.output,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            webcam=opt.webcam
        )

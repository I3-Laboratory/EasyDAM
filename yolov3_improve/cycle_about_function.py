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
from tqdm import tqdm
def generate_pesudo_label(cfg,
           weights,
           images,
           output,
           img_size=416,
           conf_thres=0.5,
           nms_thres=0.4,
           save_txt=False,
           save_images=True,
           webcam=False, conf_threshold=0.5):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch_utils.select_device()
    basepath = os.getcwd()

    if not os.path.exists(output):
        os.makedirs(output)  # make new output folder
    # Initialize model
    model = Darknet(cfg, img_size)

    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)
    model.to(device).eval()

    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size, read_txt=False)

    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])

    label_dict = OrderedDict()
    number_anchor = 0
    number_score = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    # conf_threshold = 0.9
    # global detections1
    for i, (path, img, im0) in pbar:
        t = time.time()
        save_path = str(Path(output) / Path(path).name)
        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_threshold]  # remove boxes < threshol
        """======================================="""
        try:
            if len(pred) > 0:
                detections1 = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]
            scores = detections1.data.cpu().numpy()[:, 4].tolist()
        except UnboundLocalError:
            # pass
            continue
        # continue

        number_anchor += len(scores)
        for score in scores:
            number_score += score
    print("[INFO] Number of Anchor is:", number_anchor)
    print("[INFO] Scores of all Anchor is", number_score)
    try:
        Average_Scores = number_score/number_anchor
        print("[INFO] Average Scores is:", Average_Scores)
    except ZeroDivisionError:
        pass

    pbar1 = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (path, img, im0) in pbar1:
        t = time.time()
        save_path = str(Path(output) / Path(path).name)
        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return

        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_threshold]  # remove boxes < threshol

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
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write('%g %g %g %g %g %g\n' %
                                   (x1, y1, x2, y2, cls, cls_conf * conf))
                label = ''
                if conf >= Average_Scores:
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (255,0,0), thickness=2, lineType=cv2.LINE_AA)

            detections = detections.cuda().data.cpu().numpy()
            label_list = []
            for i in range(len(detections)):
                if detections[i][5] >= Average_Scores:
                    xmin = detections[i][0]
                    ymin = detections[i][1]
                    xmax = detections[i][2]
                    ymax = detections[i][3]
                    loc = "("+str(int(xmin))+","+str(int(ymin))+","+str(int(xmax))+","+str(int(ymax))+")"
                    label_list.append(loc)
            filepath = os.path.join(basepath, path)
            label_dict[filepath] = label_list

        dt = time.time() - t
        # print('Spend time: {:.3f}s'.format(dt))

        # if save_images:  # Save generated image with detections
        #     cv2.imwrite(save_path, im0)

    if os.path.exists("data_pseudo_cycle/labels.json"):
        os.remove("data_pseudo_cycle/labels.json")
    pesudo_json = json.dumps(label_dict)
    with open("data_pseudo_cycle/labels.json", "w") as j:
        j.write(pesudo_json)
        print('[INFO] Writing labels.json finishing!!')

    label_dict = json.loads(open("data_pseudo_cycle/labels.json", 'r').read())
    if os.path.exists("data_pseudo_cycle/fruit_train.txt"):
        os.remove("data_pseudo_cycle/fruit_train.txt")
    file_txt_write = open("data_pseudo_cycle/fruit_train.txt", "w")
    for key, value in label_dict.items():
        file_txt_write.writelines(key)
        file_txt_write.writelines("\n")

    file_txt_write = open("data_pseudo_cycle/Scores.txt", "a")

    file_txt_write.writelines("Number of Anchors:" +str(number_anchor) +"\t" + "Average Anchor Scores:" + str(round(Average_Scores, 6)))
    file_txt_write.writelines("\n")
    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-orange.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--output', type=str, default='data_pseudo_cycle/output', help='path to save path')
    parser.add_argument('--images', type=str, default='data_pseudo_cycle/unlabeled_trainset/', help='path to images')
    parser.add_argument('--webcam', type=bool, default=False, help='webcam to detections or not')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    parser.add_argument('--read_txt', type=str, default=False, help='read image from test.txt')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        generate_pesudo_label(
            opt.cfg,
            opt.weights,
            opt.images,
            opt.output,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            webcam=opt.webcam,
            conf_threshold=opt.conf_thres
        )

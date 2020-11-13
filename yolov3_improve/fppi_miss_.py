# -*- coding: utf-8 -*-
# @File  : fppi_miss.py
# @Author: SmartGx
# @Date  : 19-7-1 上午11:43
# @Desc  :
import argparse

from models import *
from utils.datasets import *
from utils.utils import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def test(
        cfg,
        weights,
        data_root,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.01,
        nms_thres=0.4,
        eval=False
):
    device = torch_utils.select_device()
    # Configure run
    nC = 1

    model = Darknet(cfg, img_size)

    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    # dataloader
    if eval:
        dataloader = LoadImagesAndLabels(data_root, batch_size=batch_size, img_size=img_size, train=False, test=False)
    else:
        dataloader = LoadImagesAndLabels(data_root, batch_size=batch_size, img_size=img_size, train=False, test=True)

    if eval:
        print('[INFO] EvalSet {} samples. Start Eval....'.format(dataloader.nF))
    else:
        print('[INFO] TestSet {} samples. Start Test....'.format(dataloader.nF))

    if not eval:
        print('='*20)
        print('%11s' * 9 % ('Image', 'Total', 'P', 'R', 'F1', 'FPPI', '   Miss_Rate', 'infer_time', 'NMS_time'))
    mean_mAP, mean_R, mean_P, FPPI, Miss_Rate, seen = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    outputs, mAPs, mR, mP, TP, mFP, confidence, pred_class, target_class = [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)

    tp = []
    fp = []
    gts = 0
    name = 0
    allnum = 0

    allcorrect = []
    alldetections = []
    alltarget_cls = []

    aver_num_time = 0
    aver_test_time = 0

    for i, (rgbs, targets, img0) in enumerate(dataloader):
        gts += sum([targets[i].numpy().shape[0] for i in range(len(targets))])

        t0 = time.time()
        output = model(rgbs.to(device=device))
        t1 = time.time()
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)
        # nms_time = time.time()
        t2 = time.time()
        nms_time = (t2 - t1) / batch_size
        # print("(1)", nms_time-t1)
        aver_num_time += nms_time
        # Compute average precision for each sample
        for sample_i, (labels, detections) in enumerate(zip(targets, output)):
            seen += 1

            if detections is None:
                # If there are labels but no detections mark as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0), mFP.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections.cpu().numpy()
            detections = detections[np.argsort(-detections[:, 4])]

            # If no labels add number of detections as incorrect
            correct = []
            if labels.size(0) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mAPs.append(0), mR.append(0), mP.append(0), mFP.append(0)
                continue
            else:
                target_cls = labels[:, 0]

                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = xywh2xyxy(labels[:, 1:5]) * img_size

                detected = []
                for *pred_bbox, conf, obj_conf, obj_pred in detections:

                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes)
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                        tp.append(1)
                        fp.append(0)
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        tp.append(0)
                        fp.append(1)
                        correct.append(0)
            allcorrect.extend(correct)
            alldetections.append(detections)
            alltarget_cls.append(np.array(labels[:, 0]))
        t3 = time.time()
        test_time = (t3 - t0) / batch_size
        aver_test_time += test_time
    print("Average NMS  Cost Time is:", aver_num_time/len(dataloader))
    print("Average Test Cost Time is:", aver_test_time/len(dataloader))

    predcorrects = allcorrect
    for i,x in enumerate(alldetections):
        if i == 0:
            predscores = alldetections[0]
        else:
            predscores = np.concatenate((predscores,alldetections[i]),axis=0)
    for i,x in enumerate(alltarget_cls):
        if i == 0:
            predlabels = alltarget_cls[0]
        else:
            predlabels = np.concatenate((predlabels,alltarget_cls[i]),axis=0)


    AP, AP_class, R, P, FP, g, TP  = ap_per_class(tp=predcorrects,
                                                  conf=predscores[:, 4],
                                                  pred_cls=predscores[:, 6],
                                                  target_cls=predlabels)
    # apssd =apssd.mean()
    mAP = AP.mean()
    mR = R.mean()
    mP = P.mean()
    fppi = np.sum(FP) / dataloader.nF
    miss_rate = 1-mR
    f1 = 2.0 * (mR * mP) / (mP + mR + 0.0001)
    print('[INFO] AP: {:.4f}\tFPPI: {:.4f}\tMiss Rate: {:.4f}'.format(mAP, fppi, miss_rate))
    print('[INFO] Precision: {:.4f}\tRecall: {:.4f}\tF1-Score: {:.4f}'.format(mP, mR, f1))
    print('TPs',np.sum(TP))
    print('FPs',np.sum(FP))
    print('GTs',gts)
    print('='*60)

    with open("metric_record/" + data_root.split("/")[-1] + "_" + weights.split("/")[-1] + ".txt", "a") as file:
        if conf_thres == 0.001:
            file.write("\n")
            file.write("=" * 60 + "\n")
            file.write(("%12s" *10) % ("Conf_thresh","Precision", "Recall", "F1-Score", "AP", "FPPI", "Miss Rate", "TPs", "FPs", "GTs") + "\n")
        file.write("%10s"%(repr(conf_thres)) + '%12.4g' * 6 % (mP, mR, f1, mAP, fppi, miss_rate) + '%11s' * 3 % (repr(np.sum(TP)), repr(np.sum(FP)), repr(gts)) + '\n')

    return mAP, fppi, miss_rate, mR, mP, f1, np.sum(TP), np.sum(FP), gts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-orange.cfg', help='rgb cfg file path')
    parser.add_argument('--data_root', type=str, default='./data/all_fruit_datasets', help='Path to training set')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='0.3 is orignal , object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    with torch.no_grad():
        mAP = test(
            opt.cfg,
            opt.weights,
            opt.data_root,
            opt.batch_size,
            opt.img_size,
            opt.iou_thres,
            opt.conf_thres,
            opt.nms_thres
        )


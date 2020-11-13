
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
from utils import torch_utils
import torch.nn as nn
import math
import torchvision
# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def coco_class_weights():  # frequency of each class in coco train2014
    weights = 1 / torch.FloatTensor(
        [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380])
    weights /= weights.sum()
    return weights


def plot_one_box(x, img, color=None, label=None, line_thickness=None, point=False):  # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    center_points = ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2)
    if point:
        cv2.circle(img, center_points, 5, (0, 255, 0), -1)
    else:
        cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = torch.round(torch.clamp(coords[:, :4], min=0))
    return coords

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ssd_ap_per_class(tp, fp, num_gt):

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_gt)

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # print('-----tp------', len(tp) ,tp[-1])
    # print('-----fp------', len(fp) ,fp[-1])
    # print('-----num_gt------', num_gt)
    # print('-----rec------', rec)
    # print('-----precision-----',prec)
    ap = voc_ap(rec, prec, use_07_metric = True)

    return tp, fp, num_gt, ap, rec, prec

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))
    t_p=0
    # Create Precision-Recall curve and compute AP for each class
    ap, p, r,G = [], [], [],[]
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects
        G.append(n_gt)
        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])
            t_p+=tpc[-1]
            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))
            #print('tpc'*10,tpc[-1])
            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            # apssd = voc_ap(recall_curve, precision_curve, use_07_metric=False)
            ap.append(compute_ap(recall_curve, precision_curve))

            mrec = np.concatenate(([0.], recall_curve, [1.]))
            mpre = np.concatenate(([0.], precision_curve, [0.]))

            draw_plot(recall_curve, precision_curve, mrec, mpre)

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p), fpc[-1],np.array(G),np.array(t_p)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def draw_plot(rec,prec,mrec,mprec):
    import matplotlib.pyplot as plt
    # rec = rec.tolist()
    # prec = prec.tolist()
    #选取几个特定的点，画PR曲线
    lenrec = len(rec)
    arec=[]
    aprec=[]
    a = [0,int(lenrec//10),int(lenrec//5),int(lenrec//4),int(lenrec//3.8),int(lenrec//3.6),int(lenrec//3.4),int(lenrec//3.2),int(lenrec//3),int(lenrec//2.9),int(lenrec//2.8),int(lenrec//2.7),int(lenrec//2.6),int(lenrec//2.3),int(lenrec//2),int(lenrec//1.8),int(lenrec//1.6),int(lenrec//1.4),int(lenrec//1.2),-1]
    for i in a:
        arec.append(rec[i])
        aprec.append(prec[i])
    # print(arec,aprec)

    plt.plot(rec, prec,linewidth = 2)
      # add a new penultimate point to the list (mrec[-2], 0.0)
      # since the last line segment (and respective area) do not affect the AP value
    # area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    # area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]


    # plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
      # set window title
    fig = plt.gcf() # gcf - get current figure
    fig.canvas.set_window_title('AP ')
      # set plot title
    plt.title('class: ' + 'orange')
    #plt.suptitle('This is a somewhat long figure title', fontsize=16)
      # set axis titles
    plt.xlabel('Recall')
    plt.ylabel('Precision')
      # optional - set axes
    axes = plt.gca() # gca - get current axes
    axes.set_xlim([0.0,1.0])
    axes.set_ylim([0.0,1.05]) # .05 to give some extra space
      # Alternative option -> wait for button to be pressed
      #while not plt.waitforbuttonpress(): pass # wait for key display
      # Alternative option -> normal display
    #plt.show()
    # save the plot
    results_files_path = './'
    fig.savefig(results_files_path + 'orange' + ".png")
    plt.cla() # clear axes for next plot

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    union = (b1_area + b2_area - inter_area + 1e-16)
    iou = inter_area / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def compute_loss(p, targets, model):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets(p, targets, model)
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Compute losses
    np, ng = 0, 0  # number grid points, targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj
        np += tobj.numel()

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ng += nb
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchor_vec[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn)  # targets
                t[range(nb), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE
                # lcls += CE(ps[:, 5:], tcls[i])  # CE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    if red == 'sum':
        bs = tobj.shape[0]  # batch size
        lobj *= 3 / (6300 * bs) * 2  # 3 / np * 2
        if ng:
            lcls *= 3 / ng / model.nc
            lbox *= 3 / ng

    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()

def build_targets(target, anchor_wh, nA, nC, nG):
    """
    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    nB = len(target)  # number of images in batch
    nT = [len(x) for x in target]
    txy = torch.zeros(nB, nA, nG, nG, 2)  # batch size, anchors, grid size
    twh = torch.zeros(nB, nA, nG, nG, 2)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)  # nC = number of classes

    for b in range(nB):
        nTb = nT[b]  # number of targets
        if nTb == 0:
            continue
        t = target[b]

        gxy, gwh = t[:, 1:3] * nG, t[:, 3:5] * nG

        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        gi, gj = torch.clamp(gxy.long(), min=0, max=nG - 1).t()

        # iou of targets-anchors (using wh only)
        box1 = gwh
        box2 = anchor_wh.unsqueeze(1)
        inter_area = torch.min(box1, box2).prod(2)
        iou = inter_area / (box1.prod(1) + box2.prod(2) - inter_area + 1e-16)

        # Select best iou_pred and anchor
        iou_best, a = iou.max(0)  # best anchor [0-2] for each target

        # Select best unique target-anchor combinations
        if nTb > 1:
            iou_order = torch.argsort(-iou_best)  # best to worst

            # Unique anchor selection
            u = torch.cat((gi, gj, a), 0).view((3, -1))
            # u = torch.stack((gi, gj, a),0)
            _, first_unique = np.unique(u[:, iou_order], axis=1, return_index=True)  # first unique indices
            # _, first_unique = torch.unique(u[:, iou_order], dim=1, return_inverse=True)  # different than numpy?

            i = iou_order[first_unique]
            # best anchor must share significant commonality (iou) with target
            i = i[iou_best[i] > 0.10]  # TODO: arbitrary threshold is problematic
            if len(i) == 0:
                continue

            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:
            if iou_best < 0.10:
                continue

        tc, gxy, gwh = t[:, 0].long(), t[:, 1:3] * nG, t[:, 3:5] * nG

        # XY coordinates
        txy[b, a, gj, gi] = gxy - gxy.floor()

        # Width and height
        twh[b, a, gj, gi] = torch.log(gwh / anchor_wh[a])  # yolo method
        # twh[b, a, gj, gi] = torch.sqrt(gwh / anchor_wh[a]) / 2 # power method

        # One-hot encoding of label
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1

    return txy, twh, tconf, tcls


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        # cross-class NMS (experimental)
        cross_class_nms = False
        if cross_class_nms:
            a = pred.clone()
            _, indices = torch.sort(-a[:, 4], 0)  # sort best to worst
            a = a[indices]
            radius = 30  # area to search for cross-class ious
            for i in range(len(a)):
                if i >= len(a) - 1:
                    break

                close = (torch.abs(a[i, 0] - a[i + 1:, 0]) < radius) & (torch.abs(a[i, 1] - a[i + 1:, 1]) < radius)
                close = close.nonzero()

                if len(close) > 0:
                    close = close + i + 1
                    iou = bbox_iou(a[i:i + 1, :4], a[close.squeeze(), :4].reshape(-1, 4), x1y1x2y2=False)
                    bad = close[iou > nms_thres]

                    if len(bad) > 0:
                        mask = torch.ones(len(a)).type(torch.ByteTensor)
                        mask[bad] = 0
                        a = a[mask]
            pred = a

        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        class_prob, class_pred = torch.max(F.softmax(pred[:, 5:], 1), 1)

        v = ((pred[:, 4] > conf_thres) & (class_prob > .4))  # TODO examine arbitrary 0.4 thres here
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]
        class_prob = class_prob[v]
        class_pred = class_pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_prob, class_pred)
        detections = torch.cat((pred[:, :5], class_prob.float().unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda(prediction.device)

        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in unique_labels:
            # Get the detections with class c
            dc = detections[detections[:, -1] == c]
            # Sort the detections by maximum object confidence
            _, conf_sort_index = torch.sort(dc[:, 4] * dc[:, 5], descending=True)
            dc = dc[conf_sort_index]

            # Non-maximum suppression
            det_max = []
            if nms_style == 'OR':  # default
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(det_max[-1], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

                # Image      Total          P          R        mAP
                #  4964       5000      0.629      0.594      0.586

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[:1], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc) > 0:
                    iou = bbox_iou(dc[:1], dc[0:])  # iou with other boxes
                    i = iou > nms_thres

                    weights = dc[i, 4:5] * dc[i, 5:6]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[iou < nms_thres]

                # Image      Total          P          R        mAP
                #  4964       5000      0.633      0.598      0.589  # normal

            if len(det_max) > 0:
                det_max = torch.cat(det_max)
                # Add max detections to outputs
                output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output

def non_max_suppression_(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=True, classes=None, agnostic=False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    method = 'merge'
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * len(prediction)

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply conf constraint
        x = x[x[:, 4] > conf_thres]

        # Apply width-height constraint
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        if not torch.isfinite(x).all():
            x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # if method == 'fast_batch':
        #    x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        if method == 'merge':  # Merge NMS (boxes merged using weighted mean)
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            if n < 3E3:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                # weights = (box_iou(boxes, boxes).tril_() > iou_thres) * scores.view(-1, 1)  # box weights
                # weights /= weights.sum(0)  # normalize
                # x[:, :4] = torch.mm(weights.T, x[:, :4])
                weights = (box_iou(boxes[i], boxes) > iou_thres) * scores[None]  # box weights
                x[i, :4] = torch.mm(weights / weights.sum(1, keepdim=True), x[:, :4]).float()  # merged boxes

        elif method == 'vision':
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        elif method == 'fast':  # FastNMS from https://github.com/dbolya/yolact
            iou = box_iou(boxes, boxes).triu_(diagonal=1)  # upper triangular iou matrix
            i = iou.max(0)[0] < iou_thres

        output[xi] = x[i]
    return output

def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    import torch
    a = torch.load(filename, map_location='cpu')
    a['optimizer'] = []
    torch.save(a, filename.replace('.pt', '_lite.pt'))


def coco_class_count(path='../coco/labels/train2014/'):
    # histogram of occurrences per class
    import glob

    nC = 80  # number classes
    x = np.zeros(nC, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nC)
        print(i, len(files))


def coco_only_people(path='../coco/labels/val2014/'):
    # find images with only people
    import glob

    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def plot_results():
    # Plot YOLO training results file 'results.txt'
    import glob
    import matplotlib.pyplot as plt
    import numpy as np
    # import os; os.system('rm -rf results.txt && wget https://storage.googleapis.com/ultralytics/results_v1_0.txt')

    plt.figure(figsize=(14, 7))
    s = ['X + Y', 'Width + Height', 'Confidence', 'Classification', 'Total Loss', 'mAP', 'Recall', 'Precision']
    files = sorted(glob.glob('results*.txt'))
    for f in files:
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 9, 10, 11]).T  # column 11 is mAP
        n = results.shape[1]
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.plot(range(1, n), results[i, 1:], marker='.', label=f)
            plt.title(s[i])
            if i == 0:
                plt.legend()

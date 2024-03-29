import argparse
import time

from tensorboardX import SummaryWriter
from models import *
from utils.datasets import *
from utils.utils import *
import fppi_miss_

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train(
        cfg,
        data_root,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        multi_scale=False,
        freeze_backbone=False,
        var=0,
):
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()
    vis_model = True
    writer = SummaryWriter(log_dir='./log', comment='Train_log')

    if multi_scale:  # pass maximum multi_scale size
        img_size = 608
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Initialize model
    model = Darknet(cfg, img_size)
    print('--'*10,model)

    # Get dataloader
    dataloader = LoadImagesAndLabels(data_root, batch_size, img_size, multi_scale=multi_scale, augment=True, train=True, test=False)

    lr0 = 0.001
    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    if resume:
        checkpoint = torch.load(latest, map_location='cpu')

        # Load weights to resume from
        model.load_state_dict(checkpoint['model'])

        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        model.to(device).train()

        # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     p.requires_grad = True if (p.shape[0] == 255) else False

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)
        # optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
        #                              lr=lr0, weight_decay=1e-5)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    else:
        # Initialize model with backbone (optional)
        if cfg.endswith('yolov3.cfg') or cfg.endswith('yolov3-orange.cfg'):
            load_darknet_weights(model, weights + 'darknet53.conv.74')
            cutoff = 75
            print('[INFO] Loading pre-model [{}]....'.format(weights + 'darknet53.conv.74'))
        elif cfg.endswith('yolov3-tiny-.cfg'):
            load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
            cutoff = 15
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        model.to(device).train()
        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                                    lr=lr0, momentum=.9, weight_decay=1e-5)
        # optimizer = torch.optim.Adam(model.parameters(),
        #                             lr=lr0, weight_decay=1e-5)
    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)
    model_info(model)
    t0 = time.time()
    for epoch in range(epochs):
        epoch += start_epoch
        print(('%8s%12s' + '%10s' * 8) % (
            'Epoch', 'Batch', 'lr', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))
        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)  at 0, 54, 61 epochs to 1e-3, 1e-4, 1e-5
        # if epoch >= 0 and epoch < 50:
        #     lr = lr0 / 1
        # elif 100 > epoch >= 50:
        #     lr = lr0 / 10
        # else:
        #     lr = lr0 /100
        if epoch >= 0 and epoch < 10:
            lr = lr0 / 1
        elif 15 > epoch >= 10:
            lr = lr0 / 10
        else:
            lr = lr0 /100

        for g in optimizer.param_groups:
            g['lr'] = lr
        # 将学习率写入训练日志
        writer.add_scalar('LR', lr, epoch)
        # Freeze darknet53.conv.74 for first epoch
        if freeze_backbone and (epoch < 2):
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if (epoch == 0) else True
        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()

        for i, (imgs, targets, img0) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue
            # SGD burn-in
            if (epoch >= 0) & ((i + epoch*150) <= 600):
                lr = lr0 * ((i + epoch*150) / 600) ** 3
                for g in optimizer.param_groups:
                    g['lr'] = lr
            # Compute loss, compute gradient, update parameters
            pred = model(imgs.to(device), targets, var=var)
            loss = pred
            # loss, loss_items = compute_loss(pred, targets, model)
            loss.backward()
            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()
            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 8) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, len(dataloader) - 1),lr,
                rloss['xy'], rloss['wh'], rloss['conf'],
                rloss['cls'], rloss['loss'],
                model.losses['nT'], time.time() - t0)
            t0 = time.time()

            writer.add_scalar('Loss_total', rloss['loss'], epoch)
            print(s)

        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target
        writer.add_scalar('best_loss', best_loss, epoch)

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest)

        # Save best checkpoint
        if best_loss == loss_per_target:
            os.system('cp ' + latest + ' ' + best)

        # Save backup weights every 5 epochs (optional)
        # if (epoch >= 50) & (epoch % 50 == 0):
        #     os.system('cp ' + latest + ' ' + weights + 'backup{}.pt'.format(epoch))

        if epoch == 0:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/all_fruit_datasets',help='Path to training set')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=4, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-orange.cfg', help='cfg file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--var', type=float, default=0, help='test variable')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    train(
        opt.cfg,
        opt.data_root,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        multi_scale=opt.multi_scale,
        var=opt.var,
    )
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.DCNet import Net
from utils.dataloader_edge import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np

file = open("log_Five/DCNet.txt", "a")
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
np.random.seed(3407)
torch.backends.cudnn.benchmark = True

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)

    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den

    return loss.mean()


def train(train_loader, model, optimizer, epoch):
    model.train()
    size_rates = [1]
    loss_record4, loss_record3, loss_record2, loss_record1, loss_record4e, loss_record4t, loss_record3e, loss_record3t, loss_record2e, loss_record2t, loss_record1e, loss_record1t = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # textures = Variable(textures).cuda()
            edges = Variable(edges).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # textures = F.upsample(textures, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edges = F.upsample(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            fuse4, fuse3, fuse2, fuse1, f4_e, f4_t, f3_e, f3_t, f2_e, f2_t, f1_e, f1_t = model(images)
            # ---- loss function ----
            lossfuse4 = structure_loss(fuse4, gts)
            lossfuse3 = structure_loss(fuse3, gts)
            lossfuse2 = structure_loss(fuse2, gts)
            lossfuse1 = structure_loss(fuse1, gts)
            loss4_e = dice_loss(f4_e, edges)
            loss3_e = dice_loss(f3_e, edges)
            loss2_e = dice_loss(f2_e, edges)
            loss1_e = dice_loss(f1_e, edges)
            loss4_t = structure_loss(f4_t, gts)
            loss3_t = structure_loss(f3_t, gts)
            loss2_t = structure_loss(f2_t, gts)
            loss1_t = structure_loss(f1_t, gts)

            loss = lossfuse4 + lossfuse3 + lossfuse2 + lossfuse1 + loss4_e + loss4_t + loss3_e + loss3_t + loss2_e + loss2_t + loss1_e + loss1_t  # TODO: try different weights for loss

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record4.update(lossfuse4.data, opt.batchsize)
                loss_record3.update(lossfuse3.data, opt.batchsize)
                loss_record2.update(lossfuse2.data, opt.batchsize)
                loss_record1.update(lossfuse1.data, opt.batchsize)
                loss_record4e.update(loss4_e.data, opt.batchsize)
                loss_record3e.update(loss3_e.data, opt.batchsize)
                loss_record2e.update(loss2_e.data, opt.batchsize)
                loss_record1e.update(loss1_e.data, opt.batchsize)
                loss_record4t.update(loss4_t.data, opt.batchsize)
                loss_record3t.update(loss3_t.data, opt.batchsize)
                loss_record2t.update(loss2_t.data, opt.batchsize)
                loss_record1t.update(loss1_t.data, opt.batchsize)
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-1: {:.4f}, lateral-2: {:.4f}, lateral-3: {:.4f}, lateral-4: {:.4f}, lateral-e: {:.4f}, lateral-t: {:0.4f}, lateral-2e: {:0.4f}, lateral-2t: {:0.4f}, lateral-3e: {:0.4f}, lateral-3t: {:0.4f}, lateral-4e: {:0.4f}], lateral-4t: {:0.4f}]]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record1.avg, loss_record2.avg, loss_record3.avg, loss_record4.avg, loss_record1e.avg, loss_record1t.avg, loss_record2e.avg, loss_record2t.avg, loss_record3e.avg,
                         loss_record3t.avg, loss_record4e.avg, loss_record4t.avg))

            file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-1: {:.4f}, lateral-2: {:.4f}, lateral-3: {:.4f}, lateral-4: {:.4f}, lateral-e: {:.4f}, lateral-t: {:0.4f}, lateral-2e: {:0.4f}, lateral-2t: {:0.4f}, lateral-3e: {:0.4f}, lateral-3t: {:0.4f}, lateral-4e: {:0.4f}], lateral-4t: {:0.4f}]]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record1.avg, loss_record2.avg, loss_record3.avg, loss_record4.avg, loss_record1e.avg, loss_record1t.avg, loss_record2e.avg, loss_record2t.avg, loss_record3e.avg,
                         loss_record3t.avg, loss_record4e.avg, loss_record4t.avg))

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 1 == 0 or (epoch + 1) == opt.epoch:
        torch.save(model.state_dict(), save_path + 'Net-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'Net-%d.pth'% epoch)
        file.write('[Saving Snapshot:]' + save_path + 'Net-%d.pth' % epoch + '\n')

def test(model, epoch, opt):
    global best_mae,best_epoch

    test_data_path = 'your_path'
    save_path = 'snapshots/{}/'.format(opt.train_save)

    model.eval()

    image_root = '{}/Imgs/'.format(test_data_path)
    gt_root = '{}/GT/'.format(test_data_path)
    test_loader = test_dataset(image_root, gt_root, opt.trainsize)
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            _, _, _, res, _, _, _, _, _, _, _, _ = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum+=np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
        mae = mae_sum/test_loader.size

        if epoch==1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path+'Net_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        print('Epoch: {} MAE: {} #### bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        file.write('Epoch: {} MAE: {} #### bestMAE: {} bestEpoch: {}\n'.format(epoch,mae,best_mae,best_epoch))


best_mae=1
best_epoch=0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=384, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='your_path', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='DCNet')
    opt = parser.parse_args()

    # ---- build models ----
    model = Net().cuda()
    pre_path = 'swin_base_patch4_window12_384_22k.pth'
    model.load_pre(pre_path)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)
    # texture_root = '{}/Texture/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)
        test(model,epoch,opt)

    file.close()
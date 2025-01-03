import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from lib.DCNet import Net
from utils.dataloader_edge import test_dataset
import time

size_rates = [100]
model_name = 'DCNet'
times = []
for rate in size_rates:

    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=384, help='testing size')
    parser.add_argument('--pth_path', type=str, default='your_path/{}/Net-{}.pth'.format(model_name, rate))

    for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:

        data_path = '/your_path/{}/'.format(_data_name)
        save_path = '/your_path/{}/Net-{}.pth/{}/'.format(model_name, rate, _data_name)

        opt = parser.parse_args()
        model = Net()
        model.load_state_dict(torch.load(opt.pth_path))
        model.cuda()
        model.eval()

        os.makedirs(save_path, exist_ok=True)

        image_root = '{}/Imgs/'.format(data_path)
        gt_root = '{}/GT/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)


        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            start = time.time()

            result = model(image)
            end = time.time()
            times.append(end - start)

            res = result[3]
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imwrite(save_path+name, (res*255).astype(np.uint8))
        time_sum = 0
        for i in times:
            time_sum += i
        print("FPS: %f" % (len(times)/time_sum))


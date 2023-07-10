import argparse
import torch
import numpy as np
import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import TestSetLoader_all_twoTar
import random
from model import *

import time
import cv2

from torchvision import transforms
import os
from functools import partial
import pickle
from skimage import measure, color

parser = argparse.ArgumentParser(description="Stereo Eval")
parser.add_argument("--model_path", type=str,
                    default="",
                    help="model path")

parser.add_argument("--datafile", type=str,
                    default='testdata_sceneflow_twotarget.txt',
                    help="Txt for loading TestSet")
parser.add_argument("--scale", type=int, default=4, help="training batch size")

parser.add_argument('--save_dir', type=str,
                    default='/output/')
parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

opt = parser.parse_args()

def load_parallel(model_path):
    state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def quantize(img):
    return img.clip(0, 255).round().astype(np.uint8)


def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = measure.compare_ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s



def valid_sam(testing_data_loader, model):
    # if os.path.isfile(model_path):
    #     print("=> loading checkpoint '{}'".format(model_path))
    torch.cuda.empty_cache()
    psnr_frame = 0
    ssim_frame = 0

    y_psnr_frame = 0
    y_ssim_frame = 0

    psnr_all = 0
    ssim_all = 0

    y_psnr_all = 0
    y_ssim_all = 0

    index = 0
    frame_index = 0
    time_all = 0
    for iteration, (inputs, target_left, target_right) in enumerate(testing_data_loader):
        torch.cuda.empty_cache()
        logtext = ""
        index = index + 1

        # with torch.no_grad():
        #     inputs, target = Variable(inputs), Variable(target)
        with torch.no_grad():
            if opt.cuda:
                inputs = inputs.cuda()
                HR_left = target_left.cuda()
                # HR_right = target_left.cuda()

        with torch.no_grad():
            SR_left, SR_right = model(inputs)
        #
        # print("====>Inference time:     ", time_frame)
        # time_all = time_all + float(time_frame)


        SR_left = torch.clamp(SR_left, 0, 1)
        SR_right = torch.clamp(SR_right, 0, 1)

        SR_left_np = np.array(torch.squeeze(SR_left.data.cpu(), 0).permute(1, 2, 0))
        SR_left_np_255 = (SR_left_np * 255.0).round().astype(np.uint8)
        y_SR_left_np = quantize(color.rgb2ycbcr(SR_left_np_255)[:, :, 0])

        # SR_right_np = np.array(torch.squeeze(SR_right.data.cpu(), 0).permute(1, 2, 0))

        HR_left_np = np.array(torch.squeeze(HR_left.data.cpu(), 0).permute(1, 2, 0))
        HR_left_np_255 = (HR_left_np * 255.0).round().astype(np.uint8)
        y_HR_left_np = quantize(color.rgb2ycbcr(HR_left_np_255)[:, :, 0])
        # HR_right_up = np.array(torch.squeeze(HR_right.data.cpu(), 0).permute(1, 2, 0))

        PSNR_left = measure.compare_psnr(HR_left_np_255, SR_left_np_255)
        SSIM_left = compute_ssim(HR_left_np_255, SR_left_np_255)

        y_PSNR_left = measure.compare_psnr(y_HR_left_np, y_SR_left_np)
        y_SSIM_left = compute_ssim(y_HR_left_np, y_SR_left_np)

        # PSNR_right = measure.compare_psnr(HR_right_up, SR_right_np)
        # SSIM_right = measure.compare_ssim(HR_right_up, SR_right_np, multichannel=True)

        psnr_frame = psnr_frame + PSNR_left
        ssim_frame = ssim_frame + SSIM_left

        y_psnr_frame = y_psnr_frame + y_PSNR_left
        y_ssim_frame = y_ssim_frame + y_SSIM_left

        # psnr_epoch = psnr_epoch + PSNR_right
        # ssim_epoch = ssim_epoch + SSIM_right

        ## save results
        save_path = os.path.join(opt.save_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        HR_left_img = transforms.ToPILImage()(torch.squeeze(HR_left.data.cpu(), 0))
        HR_left_img.save(save_path + '/{:0>5d}_{:0>5d}_hr0.png'.format(frame_index, index))
        # HR_right_img = transforms.ToPILImage()(torch.squeeze(HR_right.data.cpu(), 0))
        # HR_right_img.save(save_path + '/' + scene_name + '_hr1.png')

        # SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
        # SR_left_img.save(save_path + '/{:0>5d}_{:0>5d}_sr0.png'.format(frame_index, index))
        print(
            "===>Test_Figure[{},{}]PSNR: {:.8f}dB  /  SSIM: {:.8f}dB".format(frame_index, index, PSNR_left, SSIM_left))
        logtext += "===>Test_Figure[{},{}]PSNR: {:.8f}dB  /  SSIM: {:.8f}dB".format(frame_index, index, PSNR_left,
                                                                                    SSIM_left) + "\n"

        print("===>Test_Figure[{},{}]   y_PSNR: {:.8f}dB  /  y_SSIM: {:.8f}dB".format(frame_index, index, y_PSNR_left,
                                                                                      y_SSIM_left))
        logtext += "===>Test_Figure[{},{}]   y_PSNR: {:.8f}dB  /  y_SSIM: {:.8f}dB".format(frame_index, index,
                                                                                           y_PSNR_left, y_SSIM_left) + "\n"

        # SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
        # SR_right_img.save(save_path + '/' + '/{:0>5d}_{:0>5d}_hr0.png'.format(frame_index, index))
        # print("===>Test_Figure[{}]PSNR: {:.8f}dB  /  SSIM: {:.8f}dB".format(iteration, PSNR_right, SSIM_right))

        if (iteration + 1) % 10 == 0:
            print(
                "!!!!Avg of Scene_{}:  Avg_PSNR:{:.8f} dB / Avg_SSIM: {:.8f} dB".format(frame_index, psnr_frame / 10,
                                                                                        ssim_frame / 10))
            logtext += "!!!!Avg_of Scene_{}:  Avg_PSNR: {:.8f} dB / Avg_SSIM: {:.8f} dB".format(frame_index,
                                                                                                psnr_frame / 10,
                                                                                                ssim_frame / 10) + "\n"

            print(
                "!!!!Avg of Scene_{}:  Avg_PSNR:{:.8f} dB / Avg_SSIM: {:.8f} dB".format(frame_index, y_psnr_frame / 10,
                                                                                        y_ssim_frame / 10))
            logtext += "!!!!Avg_of Scene_{}:  Avg_PSNR: {:.8f} dB / Avg_SSIM: {:.8f} dB".format(frame_index,
                                                                                                y_psnr_frame / 10,
                                                                                                y_ssim_frame / 10) + "\n"

            psnr_all = psnr_all + psnr_frame / 10
            ssim_all = ssim_all + ssim_frame / 10

            y_psnr_all = y_psnr_all + y_psnr_frame / 10
            y_ssim_all = y_ssim_all + y_ssim_frame / 10

            psnr_frame = 0
            ssim_frame = 0

            y_psnr_frame = 0
            y_ssim_frame = 0

            index = 0
            frame_index = frame_index + 1
        with open(os.path.join(save_path, "test_log.txt"), 'a') as f:
            f.write(logtext + "\n")

    print(
        "!!!Avg_all:  Avg_PSNR:{:.8f} dB / Avg_SSIM: {:.8f} dB".format(psnr_all / ((iteration + 1) / 10),
                                                                       ssim_all / ((iteration + 1) / 10)))
    logtext += "!!!Avg_all:  Avg_PSNR: {:.8f} dB / Avg_SSIM: {:.8f} dB".format(psnr_all / ((iteration + 1) / 10),
                                                                               ssim_all / ((iteration + 1) / 10)) + "\n"

    print(
        "!!!Avg_all:  Avg_PSNR:{:.8f} dB / Avg_SSIM: {:.8f} dB".format(y_psnr_all / ((iteration + 1) / 10),
                                                                       y_ssim_all / ((iteration + 1) / 10)))
    logtext += "!!!Avg_all:  Avg_PSNR: {:.8f} dB / Avg_SSIM: {:.8f} dB".format(y_psnr_all / ((iteration + 1) / 10),
                                                                               y_ssim_all / (
                                                                                           (iteration + 1) / 10)) + "\n"
    print(
        "!!!Avg_Time".format(time_all / ((iteration + 1))))
    logtext += "!!!Avg_Time".format(time_all / ((iteration + 1))) + "\n"

    with open(os.path.join(save_path, "test_log.txt"), 'a') as f:
        f.write(logtext + "\n")





def main(opt):
    torch.cuda.empty_cache()
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = opt.model_path

    model = SVSRNet(opt)
    # model = torch.nn.DataParallel(model)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
    #     model_dict = model.state_dict()
    #     loaded_dict = torch.load(model_path)
    #     loaded_dict = {k: v for k, v in loaded_dict.items() if k in model_dict}
    #     model_dict.update(loaded_dict)
    #     model.load_state_dict(model_dict)
        model.load_state_dict(load_parallel(model_path))

    # model_path = opt.model_path
    # model = torch.load(model_path)
    # model = torch.nn.DataParallel(model, device_ids=[0])
    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module
    model.eval()

    if opt.cuda:
        model.cuda()

    test_set = TestSetLoader_all_twoTar(opt)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

    oldtime = datetime.datetime.now()
    valid_sam(test_loader, model)
    newtime = datetime.datetime.now()
    print('Time consuming: ', newtime - oldtime)



if __name__ == '__main__':
    if opt.cuda:
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    print(opt.model_path)
    opt.in_ch = 3
    opt.scale = 4
    opt.embed_ch = 64

    opt.nframes = 3*2
    opt.groups = 8
    opt.front_RBs = 5
    opt.back_RBs = 10

    print(opt)
    main(opt)

# -*- coding: utf-8 -*
# !/usr/local/bin/python
import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import *

import utils
import losses
from arch_util import flow_warping
from model import *
from skimage import measure

from raft.raft import *

# Training settings
parser = argparse.ArgumentParser(description="StereoSR")
# data loader
parser.add_argument("--datafile", type=str,
                    default='traindata_sceneflow_twotarget_warploss.txt',
                        help="Txt for loading TrainSet")
parser.add_argument("--scale", type=int, default=4, help="training batch size")
parser.add_argument('--augmentation', type=bool, default=True, help='prefix of different dataset')
parser.add_argument('--patch_size', type=int, default=64, help='file for labels')
parser.add_argument('--hflip', type=bool, default=True, help='prefix of different dataset')
parser.add_argument('--rot', type=bool, default=True, help='prefix of different dataset')

# model&events path
parser.add_argument('--log_path', default='', help='log path')
parser.add_argument('--model_path', default='', help='model')
parser.add_argument('--model_name', default='', help='model')

# train setting
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--start-epoch", default=5, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--lr", type=float, default=1 * 1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument('--gamma', type=float, default=0.5, help='Learning Rate decay')
parser.add_argument("--step", type=int, default=40,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")

parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')


def main(arg):
    # 保证每次初始化一致
    # arg.seed = random.randint(1, 10000)
    # print("Random Seed: ", arg.seed)
    torch.manual_seed(arg.seed)

    cuda = arg.cuda
    if cuda:
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
        torch.cuda.manual_seed(arg.seed)
    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = utils.TrainSetLoader_all_twoTarget_warploss(arg)
    training_data_loader = DataLoader(dataset=train_set, num_workers=arg.threads, batch_size=arg.batchSize,
                                      shuffle=True)

    print("===> Building model")
    arg.cuda = torch.cuda.is_available()
    model = SVSRNet(arg)

    criterion = nn.MSELoss()
    criterionCharb = losses.CharbonnierLoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        criterion = criterion.cuda()
        criterionCharb.cuda()

    load_model_path = ""
    if os.path.isfile(load_model_path):
        print("=> loading checkpoint '{}'".format(load_model_path))
        model.load_state_dict(torch.load(load_model_path))

    ### Load pretrained RAFT
    arg.small = False
    arg.mixed_precision = False
    arg.alternate_corr = False

    FlowNet = RAFT(arg).cuda()
    flownet_modelpath = "./raft_things_reload.pth"
    print("===> Load %s" % flownet_modelpath)
    FlowNet.load_state_dict(torch.load(flownet_modelpath))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=arg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg.step, gamma=arg.gamma)

    print("===> Training")
    event_dir = os.path.join(arg.log_path, arg.model_name, 'event')
    print("===> event dir", event_dir)
    event_writer = SummaryWriter(event_dir)

    model_out_path = os.path.join(arg.model_path, arg.model_name)
    print("===> model_path", model_out_path)
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)
    print()

    Best = 0
    total_iter = 0
    for epoch in range(arg.start_epoch, arg.nEpochs + 1):
        loss_epoch = 0
        psnr_epoch = 0
        model.train()
        scheduler.step()
        print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
        for iteration, (inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3) in enumerate(training_data_loader):

            total_iter = total_iter + 1
            # inputs, target_left, target_right = Variable(inputs), Variable(target_left), Variable(target_right)

            if arg.cuda:
                inputs_1 = inputs_1.cuda()
                target_left_1 = target_left_1.cuda()
                target_right_1 = target_right_1.cuda()

                inputs_2 = inputs_2.cuda()
                target_left_2 = target_left_2.cuda()
                target_right_2 = target_right_2.cuda()

                inputs_3 = inputs_3.cuda()
                target_left_3 = target_left_3.cuda()
                target_right_3 = target_right_3.cuda()

            pred_left_1, pred_right_1 = model(inputs_1)
            pred_left_2, pred_right_2 = model(inputs_2)
            pred_left_3, pred_right_3 = model(inputs_3)

            loss_left_1 = criterionCharb(pred_left_1, target_left_1)
            loss_right_1 = criterionCharb(pred_right_1, target_right_1)
            loss_1 = loss_left_1 + loss_right_1

            loss_left_2 = criterionCharb(pred_left_2, target_left_2)
            loss_right_2 = criterionCharb(pred_right_2, target_right_2)
            loss_2 = loss_left_2 + loss_right_2

            loss_left_3 = criterionCharb(pred_left_3, target_left_3)
            loss_right_3 = criterionCharb(pred_right_3, target_right_3)
            loss_3 = loss_left_3 + loss_right_3

            loss_charb = loss_1 + loss_2 + loss_3




            ## submodule
            # flow_warping = Resample2d().cuda()

            ###  Temporal Consistency loss
            ###### left 1 frame to 2 frame
            _, flow_l_21 = FlowNet(target_left_2, target_left_1,iters=20, test_mode=True)   # flow 2 frame->1 frame, used for warp 1 frame to 2 frame
            ### warp
            warp_tar_left_1 = flow_warping(target_left_1, flow_l_21)
            warp_pre_left_1 = flow_warping(pred_left_1, flow_l_21)
            ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
            noc_mask_l_21 = torch.exp(-50 * torch.sum(target_left_2 - warp_tar_left_1, dim=1).pow(2)).unsqueeze(1)
            ST_loss_l_21 = criterion(pred_left_2 * noc_mask_l_21, warp_pre_left_1 * noc_mask_l_21)

            ###### right 1 frame to 2 frame
            _, flow_r_21 = FlowNet(target_right_2, target_right_1,iters=20, test_mode=True)  # flow 2 frame->1 frame, used for warp 1 frame to 2 frame
            ### warp
            warp_tar_right_1 = flow_warping(target_right_1, flow_r_21)
            warp_pre_right_1 = flow_warping(pred_right_1, flow_r_21)
            ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
            noc_mask_r_21 = torch.exp(-50 * torch.sum(target_right_2 - warp_tar_right_1, dim=1).pow(2)).unsqueeze(1)
            ST_loss_r_21 = criterion(pred_right_2 * noc_mask_r_21, warp_pre_right_1 * noc_mask_r_21)

            ###### left 2 frame to 3 frame
            _, flow_l_32 = FlowNet(target_left_3, target_left_2, iters=20, test_mode=True)  # flow 3 frame->2 frame, used for warp 2 frame to 3 frame
            ### warp
            warp_tar_left_2 = flow_warping(target_left_2, flow_l_32)
            warp_pre_left_2 = flow_warping(pred_left_2, flow_l_32)
            ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
            noc_mask_l_32 = torch.exp(-50 * torch.sum(target_left_3 - warp_tar_left_2, dim=1).pow(2)).unsqueeze(1)
            ST_loss_l_32 = criterion(pred_left_3 * noc_mask_l_32, warp_pre_left_2 * noc_mask_l_32)

            ###### right 2 frame to 3 frame
            _, flow_r_32 = FlowNet(target_right_3, target_right_2,iters=20, test_mode=True)  # flow 3 frame->2 frame, used for warp 2 frame to 3 frame
            ### warp
            warp_tar_right_2 = flow_warping(target_right_2, flow_r_32)
            warp_pre_right_2 = flow_warping(pred_right_2, flow_r_32)
            ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
            noc_mask_r_32 = torch.exp(-50 * torch.sum(target_right_3 - warp_tar_right_2, dim=1).pow(2)).unsqueeze(1)
            ST_loss_r_32 = criterion(pred_right_3 * noc_mask_r_32, warp_pre_right_2 * noc_mask_r_32)

            ### View Consistency loss
            ###### right 1 frame to left 1 frame
            _, flow_left2right_1 = FlowNet(target_left_1, target_right_1,iters=20, test_mode=True)  # flow left frame->right frame, used for warp right frame to left frame
            ### warp
            warp_tar_right2left_1 = flow_warping(target_right_1, flow_left2right_1)
            warp_pre_right2left_1 = flow_warping(pred_right_1, flow_left2right_1)
            ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
            noc_mask_right2left_1 = torch.exp(-50 * torch.sum(target_left_1 - warp_tar_right2left_1, dim=1).pow(2)).unsqueeze(1)
            ST_loss_right2left_1 = criterion(pred_left_1 * noc_mask_right2left_1, warp_pre_right2left_1 * noc_mask_right2left_1)

            ###### right 2 frame to left 2 frame
            _, flow_left2right_2 = FlowNet(target_left_2, target_right_2,iters=20, test_mode=True)  # flow left frame->right frame, used for warp right frame to left frame
            ### warp
            warp_tar_right2left_2 = flow_warping(target_right_2, flow_left2right_2)
            warp_pre_right2left_2 = flow_warping(pred_right_2, flow_left2right_2)
            ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
            noc_mask_right2left_2 = torch.exp(-50 * torch.sum(target_left_2 - warp_tar_right2left_2, dim=1).pow(2)).unsqueeze(1)
            ST_loss_right2left_2 = criterion(pred_left_2 * noc_mask_right2left_2, warp_pre_right2left_2 * noc_mask_right2left_2)

            ###### right 3 frame to left 3 frame
            _, flow_left2right_3 = FlowNet(target_left_3, target_right_3,iters=20, test_mode=True)  # flow left frame->right frame, used for warp right frame to left frame
            ### warp
            warp_tar_right2left_3 = flow_warping(target_right_3, flow_left2right_3)
            warp_pre_right2left_3 = flow_warping(pred_right_3, flow_left2right_3)
            ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
            noc_mask_right2left_3 = torch.exp(-50 * torch.sum(target_left_3 - warp_tar_right2left_3, dim=1).pow(2)).unsqueeze(1)
            ST_loss_right2left_3 = criterion(pred_left_3 * noc_mask_right2left_3, warp_pre_right2left_3 * noc_mask_right2left_3)

            ###  View-Temporal Consistency loss
            ###### right 1 frame to left 2 frame
            _, flow_left2right_21 = FlowNet(target_left_2, target_right_1,iters=20, test_mode=True)  # flow left 2 frame-> right 1 frame, used for warp right 1 frame to left 2 frame
            ### warp
            warp_tar_right2left_1to2 = flow_warping(target_right_1, flow_left2right_21)
            warp_pre_right2left_1to2 = flow_warping(pred_right_1, flow_left2right_21)
            ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
            noc_mask_right2left_1to2 = torch.exp(-50 * torch.sum(target_left_2 - warp_tar_right2left_1to2, dim=1).pow(2)).unsqueeze(1)
            ST_loss_right2left_1to2 = criterion(pred_left_2 * noc_mask_right2left_1to2, warp_pre_right2left_1to2 * noc_mask_right2left_1to2)

            ###### right 1 frame to left 2 frame
            _, flow_left2right_23 = FlowNet(target_left_2, target_right_3,iters=20, test_mode=True)  # flow left 2 frame-> right 3 frame, used for warp right 3 frame to left 2 frame
            ### warp
            warp_tar_right2left_3to2 = flow_warping(target_right_3, flow_left2right_23)
            warp_pre_right2left_3to2 = flow_warping(pred_right_3, flow_left2right_23)
            ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
            noc_mask_right2left_3to2 = torch.exp(-50 * torch.sum(target_left_2 - warp_tar_right2left_3to2, dim=1).pow(2)).unsqueeze(1)
            ST_loss_right2left_3to2 = criterion(pred_left_2 * noc_mask_right2left_3to2, warp_pre_right2left_3to2 * noc_mask_right2left_3to2)

            ###### left 1 frame to right 2 frame
            _, flow_right2left_21 = FlowNet(target_right_2, target_left_1,iters=20, test_mode=True)  # flow right 2 frame-> left 1 frame, used for warp left 1 frame to right 2 frame
            ### warp
            warp_tar_left2right_1to2 = flow_warping(target_left_1, flow_right2left_21)
            warp_pre_left2right_1to2 = flow_warping(pred_left_1, flow_right2left_21)
            ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
            noc_mask_right2left_1to2 = torch.exp(-50 * torch.sum(target_right_2 - warp_tar_left2right_1to2, dim=1).pow(2)).unsqueeze(1)
            ST_loss_left2right_1to2 = criterion(pred_right_2 * noc_mask_right2left_1to2, warp_pre_left2right_1to2 * noc_mask_right2left_1to2)

            ###### left 3 frame to right 2 frame
            _, flow_right2left_23 = FlowNet(target_right_2, target_left_3,iters=20, test_mode=True)  # flow right 2 frame-> left 3 frame, used for warp left 3 frame to right 2 frame
            ### warp
            warp_tar_left2right_3to2 = flow_warping(target_left_3, flow_right2left_23)
            warp_pre_left2right_3to2 = flow_warping(pred_left_3, flow_right2left_23)
            ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
            noc_mask_right2left_3to2 = torch.exp(-50 * torch.sum(target_right_2 - warp_tar_left2right_3to2, dim=1).pow(2)).unsqueeze(1)
            ST_loss_left2right_3to2 = criterion(pred_right_2 * noc_mask_right2left_3to2, warp_pre_left2right_3to2 * noc_mask_right2left_3to2)

            loss_cons = ST_loss_l_21 + ST_loss_l_32 + ST_loss_r_21 + ST_loss_r_32 + \
                        ST_loss_right2left_1 + ST_loss_right2left_2 + ST_loss_right2left_3 +\
                        ST_loss_right2left_1to2 + ST_loss_right2left_3to2 + ST_loss_left2right_1to2 + ST_loss_left2right_3to2

            loss = loss_charb + 0.001 * loss_cons



            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction_left_2 = torch.clamp(pred_left_2, 0.0, 1.0)
            prediction_right_2 = torch.clamp(pred_right_2, 0.0, 1.0)

            psnr_iter = (cal_psnr(target_left_2, pred_left_2) + cal_psnr(target_right_2, pred_right_2))/2

            psnr_epoch += psnr_iter
            loss_epoch += loss.item()

            if iteration % 50 == 0:
                print(
                    "===> Epoch[{}] Iteration[{}]: Loss: {:.5f} PSNR: {:.5f} | Loss_charb: {:.5f} Loss_cons: {:.5f} ".format(
                        epoch, iteration, loss.item(),
                        psnr_iter, loss_charb.item(), loss_cons.item()))
            if total_iter % 50 == 0:
                event_writer.add_scalar('Loss', loss.item(), total_iter)
                event_writer.add_scalar('PSNR', psnr_iter, total_iter)

            if total_iter % 500 == 0:
                event_writer.add_image('Prediction', prediction_left_2[0, :, :, :], total_iter)
                event_writer.add_image('Prediction', prediction_right_2[0, :, :, :], total_iter)

                event_writer.add_image('target', target_left_2[0, :, :, :], total_iter)
                event_writer.add_image('target', target_right_2[0, :, :, :], total_iter)

        is_best = psnr_epoch > Best
        Best = max(psnr_epoch, Best)
        if is_best or epoch % 1 == 0:
            model_save = os.path.join(model_out_path, "model_epoch_{}.pth".format(epoch))
            torch.save(model.state_dict(), model_save)
            print("Checkpoint saved to {}".format(model_save))

        # adjust_learning_rate(optimizer, epoch, loss_epoch / (iteration + 1))
        print("===> Epoch[{}]|PSNR: {:.5f}|loss: {:.5f}|Best: {:.5f}".format(epoch, psnr_epoch / (iteration + 1),
                                                                             loss_epoch / (iteration + 1),
                                                                             Best / (iteration + 1)))


def cal_psnr(img1, img2):
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1_np = img1.detach().numpy()
    img2_np = img2.detach().numpy()
    return measure.compare_psnr(img1_np, img2_np)


if __name__ == "__main__":
    arg = parser.parse_args()
    arg.in_ch = 3
    arg.scale = 4
    arg.embed_ch = 64

    arg.nframes = 3 * 2
    arg.groups = 8
    arg.front_RBs = 5
    arg.back_RBs = 10

    print(arg)
    main(arg)

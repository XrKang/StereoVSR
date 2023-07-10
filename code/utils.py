from PIL import Image, ImageOps
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor
import random
import torch
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_patch(img_in_left, img_in_right, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in_left[0].size

    ip = patch_size  # input_patch_size
    tp = scale * patch_size  # target_patch_size

    # randomly crop
    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    img_in_right = img_in_right.crop((iy, ix, iy + ip, ix + ip))
    img_in_left = [j.crop((iy, ix, iy + ip, ix + ip)) for j in img_in_left]  # [:, iy:iy + ip, ix:ix + ip]

    return img_in_left, img_in_right, img_tar


def get_patch_all(inputs, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = inputs[0].size

    ip = patch_size  # input_patch_size
    tp = scale * patch_size  # target_patch_size

    # randomly crop
    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    img_in = [j.crop((iy, ix, iy + ip, ix + ip)) for j in inputs]  # [:, iy:iy + ip, ix:ix + ip]

    return img_in, img_tar


def get_patch_twoTarget(inputs, left_tar, right_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = inputs[0].size

    ip = patch_size  # input_patch_size
    tp = scale * patch_size  # target_patch_size

    # randomly crop
    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    left_tar = left_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    right_tar = right_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]

    img_in = [j.crop((iy, ix, iy + ip, ix + ip)) for j in inputs]  # [:, iy:iy + ip, ix:ix + ip]

    return img_in, left_tar, right_tar


def get_patch_twoTarget_warploss(inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = inputs_1[0].size

    ip = patch_size  # input_patch_size
    tp = scale * patch_size  # target_patch_size

    # randomly crop
    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    target_left_1 = target_left_1.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    target_right_1 = target_right_1.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]

    target_left_2 = target_left_2.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    target_right_2 = target_right_2.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]

    target_left_3 = target_left_3.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    target_right_3 = target_right_3.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]

    img_in_1 = [j.crop((iy, ix, iy + ip, ix + ip)) for j in inputs_1]  # [:, iy:iy + ip, ix:ix + ip]
    img_in_2 = [j.crop((iy, ix, iy + ip, ix + ip)) for j in inputs_2]  # [:, iy:iy + ip, ix:ix + ip]
    img_in_3 = [j.crop((iy, ix, iy + ip, ix + ip)) for j in inputs_3]  # [:, iy:iy + ip, ix:ix + ip]

    return img_in_1, img_in_2, img_in_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3




def augment(img_in_left, img_in_right, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_tar = ImageOps.flip(img_tar)
        img_in_right = ImageOps.flip(img_in_right)
        img_in_left = [ImageOps.flip(j) for j in img_in_left]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_tar = ImageOps.mirror(img_tar)
            img_in_right = ImageOps.mirror(img_in_right)
            img_in_left = [ImageOps.mirror(j) for j in img_in_left]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_tar = img_tar.rotate(180)
            img_in_right = img_in_right.rotate(180)
            img_in_left = [j.rotate(180) for j in img_in_left]
            info_aug['trans'] = True

    return img_in_left, img_in_right, img_tar


def augment_all(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_tar = ImageOps.flip(img_tar)
        img_in = [ImageOps.flip(j) for j in img_in]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_tar = ImageOps.mirror(img_tar)
            img_in = [ImageOps.mirror(j) for j in img_in]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_tar = img_tar.rotate(180)
            img_in = [j.rotate(180) for j in img_in]
            info_aug['trans'] = True

    return img_in, img_tar


def augment_twoTarget(img_in, left_tar, right_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        left_tar = ImageOps.flip(left_tar)
        right_tar = ImageOps.flip(right_tar)

        img_in = [ImageOps.flip(j) for j in img_in]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            left_tar = ImageOps.mirror(left_tar)
            right_tar = ImageOps.mirror(right_tar)

            img_in = [ImageOps.mirror(j) for j in img_in]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            left_tar = left_tar.rotate(180)
            right_tar = right_tar.rotate(180)

            img_in = [j.rotate(180) for j in img_in]
            info_aug['trans'] = True

    return img_in, left_tar, right_tar


def augment_twoTarget_warploss(img_in_1, img_in_2, img_in_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        target_left_1 = ImageOps.flip(target_left_1)
        target_right_1 = ImageOps.flip(target_right_1)
        img_in_1 = [ImageOps.flip(j) for j in img_in_1]

        target_left_2 = ImageOps.flip(target_left_2)
        target_right_2 = ImageOps.flip(target_right_2)
        img_in_2 = [ImageOps.flip(j) for j in img_in_2]

        target_left_3 = ImageOps.flip(target_left_3)
        target_right_3 = ImageOps.flip(target_right_3)
        img_in_3 = [ImageOps.flip(j) for j in img_in_3]

        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:

            target_left_1 = ImageOps.mirror(target_left_1)
            target_right_1 = ImageOps.mirror(target_right_1)
            img_in_1 = [ImageOps.mirror(j) for j in img_in_1]

            target_left_2 = ImageOps.mirror(target_left_2)
            target_right_2 = ImageOps.mirror(target_right_2)
            img_in_2 = [ImageOps.mirror(j) for j in img_in_2]

            target_left_3 = ImageOps.mirror(target_left_3)
            target_right_3 = ImageOps.mirror(target_right_3)
            img_in_3 = [ImageOps.mirror(j) for j in img_in_3]

            info_aug['flip_v'] = True

    return img_in_1, img_in_2, img_in_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3



def get_image(img):
    img = Image.open(img).convert('RGB')
    return img


def load_image_train_all(group):
    images = [get_image(img) for img in group]
    inputs = images[:-1]
    target = images[-1]

    return inputs, target


def load_image_train_twoTarget(group):
    images = [get_image(img) for img in group]
    inputs = images[:-2]
    target_left = images[-2]
    target_right = images[-1]

    return inputs, target_left, target_right


def load_image_train_twoTarget_warploss(group):
    images = [get_image(img) for img in group]
    inputs_left_1 = images[:3]  # 0, 1, 2
    inputs_left_2 = images[1:4]  # 1, 2, 3
    inputs_left_3 = images[2:5]  # 2, 3, 4

    inputs_right_1 = images[5:8]  # 5 , 6, 7
    inputs_right_2 = images[6:9]  # 6, 7, 8
    inputs_right_3 = images[7:10]  # 7, 8, 9

    inputs_1 = inputs_left_1 + inputs_right_1
    inputs_2 = inputs_left_2 + inputs_right_2
    inputs_3 = inputs_left_3 + inputs_right_3

    target_left_1 = images[10]
    target_left_2 = images[11]
    target_left_3 = images[12]

    target_right_1 = images[13]
    target_right_2 = images[14]
    target_right_3 = images[15]

    return inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3


def load_image_train_twoTarget_warploss2(group):
    images = [get_image(img) for img in group]
    inputs_left_1 = images[:3]  # 0, 1, 2
    inputs_left_2 = images[1:4]  # 1, 2, 3
    # inputs_left_3 = images[2:5]  # 2, 3, 4

    inputs_right_1 = images[5:8]  # 5 , 6, 7
    inputs_right_2 = images[6:9]  # 6, 7, 8
    # inputs_right_3 = images[7:10]  # 7, 8, 9

    inputs_1 = inputs_left_1 + inputs_right_1
    inputs_2 = inputs_left_2 + inputs_right_2
    # inputs_3 = inputs_left_3 + inputs_right_3

    target_left_1 = images[10]
    target_left_2 = images[11]
    # target_left_3 = images[12]

    target_right_1 = images[13]
    target_right_2 = images[14]
    # target_right_3 = images[15]

    return inputs_1, inputs_2, target_left_1, target_right_1, target_left_2, target_right_2



def load_image_train2(group):
    images = [get_image(img) for img in group]
    inputs_left = images[:-2]
    input_right = images[-2]
    target = images[-1]

    return inputs_left, input_right, target


def load_image_train_test(group, scale):
    print(group[-1])

    images = [get_image(img) for img in group]

    inputs_left = images[:-2]
    input_right = images[-2]
    target = images[-1]

    w, h = inputs_left[1].size  # w,h和array是相反的
    w = (w // 4) * 4
    h = (h // 4) * 4

    target = np.array(target, dtype=np.float32)
    target = target[0:h * scale, 0:w * scale, :]

    inputs_left = [np.array(i, dtype=np.float32) for i in inputs_left]
    inputs_left_new = []
    for img_lr_left in inputs_left:
        img_lr_left = img_lr_left[0:h, 0:w, :]
        img_lr_left = Image.fromarray(img_lr_left.astype('uint8')).convert('RGB')
        inputs_left_new.append(img_lr_left)

    input_right = np.array(input_right, dtype=np.float32)
    input_right = input_right[0:h, 0:w, :]

    target = Image.fromarray(target.astype('uint8')).convert('RGB')
    input_right = Image.fromarray(input_right.astype('uint8')).convert('RGB')

    return inputs_left_new, input_right, target


def load_image_train_test_right(group, scale):
    print(group[-3])

    images = [get_image(img) for img in group]

    inputs_left = images[:3]
    input_right = images[3:6]
    target = images[6]

    w, h = input_right[1].size  # w,h和array是相反的
    w = (w // 4) * 4
    h = (h // 4) * 4

    target = np.array(target, dtype=np.float32)
    target = target[0:h * scale, 0:w * scale, :]

    input_right = [np.array(i, dtype=np.float32) for i in input_right]
    input_right_new = []
    for img_lr_right in input_right:
        img_lr_right = img_lr_right[0:h, 0:w, :]
        img_lr_right = Image.fromarray(img_lr_right.astype('uint8')).convert('RGB')
        input_right_new.append(img_lr_right)

    # inputs_left = np.array(inputs_left, dtype=np.float32)
    # input_right = input_right[0:h, 0:w, :]

    target = Image.fromarray(target.astype('uint8')).convert('RGB')
    # input_right = Image.fromarray(input_right.astype('uint8')).convert('RGB')

    return input_right_new, input_right_new, target


def load_image_train_all_test(group, scale):
    print(group[-1])

    images = [get_image(img) for img in group]
    inputs = images[:-1]
    target = images[-1]

    w, h = inputs[1].size  # w,h和array是相反的
    w = (w // 4) * 4
    h = (h // 4) * 4

    target = np.array(target, dtype=np.float32)
    target = target[0:h * scale, 0:w * scale, :]

    inputs = [np.array(i, dtype=np.float32) for i in inputs]
    inputs_new = []
    for img_inputs in inputs:
        img_lr = img_inputs[0:h, 0:w, :]
        img_lr = Image.fromarray(img_lr.astype('uint8')).convert('RGB')
        inputs_new.append(img_lr)

    target = Image.fromarray(target.astype('uint8')).convert('RGB')

    return inputs_new, target


def load_image_test_twoTarget(group, scale):
    print(group[-1])

    images = [get_image(img) for img in group]
    inputs = images[:-2]
    target_left = images[-2]
    target_right = images[-1]

    w, h = inputs[1].size  # w,h和array是相反的
    w = (w // 4) * 4
    h = (h // 4) * 4

    target_left = np.array(target_left, dtype=np.float32)
    target_left = target_left[0:h * scale, 0:w * scale, :]

    target_right = np.array(target_right, dtype=np.float32)
    target_right = target_right[0:h * scale, 0:w * scale, :]

    inputs = [np.array(i, dtype=np.float32) for i in inputs]
    inputs_new = []
    for img_inputs in inputs:
        img_lr = img_inputs[0:h, 0:w, :]
        img_lr = Image.fromarray(img_lr.astype('uint8')).convert('RGB')
        inputs_new.append(img_lr)

    target_left = Image.fromarray(target_left.astype('uint8')).convert('RGB')
    target_right = Image.fromarray(target_right.astype('uint8')).convert('RGB')

    return inputs_new, target_left, target_right


def load_image_all_test_crop(group, scale):
    print(group[-1])

    images = [get_image(img) for img in group]
    inputs = images[:-1]
    target = images[-1]

    # w, h = inputs[1].size  # w,h和array是相反的
    # w = (w // 4) * 4
    # h = (h // 4) * 4

    target = np.array(target, dtype=np.float32)
    target = target[0:128 * scale, 0:64 * scale, :]

    inputs = [np.array(i, dtype=np.float32) for i in inputs]
    inputs_new = []
    for img_inputs in inputs:
        img_lr = img_inputs[0:128, 0:64, :]
        img_lr = Image.fromarray(img_lr.astype('uint8')).convert('RGB')
        inputs_new.append(img_lr)

    target = Image.fromarray(target.astype('uint8')).convert('RGB')

    return inputs_new, target



def transform():
    return Compose([
        ToTensor(),
    ])


class TrainSetLoader(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TrainSetLoader, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale
        self.patch_size = opt.patch_size
        self.data_augmentation = opt.augmentation
        self.hflip = opt.hflip
        self.rot = opt.rot

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):

        inputs_lefts, input_right, target = load_image_train2(self.image_filenames[index])

        if self.patch_size != 0:
            inputs_lefts, input_right, target = get_patch(inputs_lefts, input_right, target, self.patch_size,
                                                          self.upscale_factor)

        if self.data_augmentation:
            inputs_lefts, input_right, target = augment(inputs_lefts, input_right, target, self.hflip, self.rot)

        if self.transform:
            target = self.transform(target)
            input_right = self.transform(input_right)
            inputs_lefts = [self.transform(j) for j in inputs_lefts]

        inputs_lefts = torch.cat((torch.unsqueeze(inputs_lefts[0], 0), torch.unsqueeze(inputs_lefts[1], 0),
                                  torch.unsqueeze(inputs_lefts[2], 0)))

        return inputs_lefts, input_right, target

    def __len__(self):
        return len(self.image_filenames)


class TrainSetLoader_all(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TrainSetLoader_all, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale
        self.patch_size = opt.patch_size
        self.data_augmentation = opt.augmentation
        self.hflip = opt.hflip
        self.rot = opt.rot

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):

        inputs, target = load_image_train_all(self.image_filenames[index])

        if self.patch_size != 0:
            inputs, target = get_patch_all(inputs, target, self.patch_size,
                                           self.upscale_factor)

        if self.data_augmentation:
            inputs, target = augment_all(inputs, target, self.hflip, self.rot)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0)))

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)


class TrainSetLoader_all_twoTarget(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TrainSetLoader_all_twoTarget, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale
        self.patch_size = opt.patch_size
        self.data_augmentation = opt.augmentation
        self.hflip = opt.hflip
        self.rot = opt.rot

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):

        inputs, target_left, target_right = load_image_train_twoTarget(self.image_filenames[index])

        if self.patch_size != 0:
            inputs, target_left, target_right = get_patch_twoTarget(inputs, target_left, target_right, self.patch_size,
                                                                    self.upscale_factor)

        if self.data_augmentation:
            inputs, target_left, target_right = augment_twoTarget(inputs, target_left, target_right, self.hflip, self.rot)

        if self.transform:
            target_left = self.transform(target_left)
            target_right = self.transform(target_right)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0)))

        return inputs, target_left, target_right

    def __len__(self):
        return len(self.image_filenames)


class TrainSetLoader_all_twoTarget_warploss(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TrainSetLoader_all_twoTarget_warploss, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale
        self.patch_size = opt.patch_size
        self.data_augmentation = opt.augmentation
        self.hflip = opt.hflip
        self.rot = opt.rot

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):

        inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3 = \
            load_image_train_twoTarget_warploss(self.image_filenames[index])

        if self.patch_size != 0:
            inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3 =\
                get_patch_twoTarget_warploss(inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2,
                                             target_right_2, target_left_3, target_right_3, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3\
                = augment_twoTarget_warploss(inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2,
                                             target_right_2, target_left_3, target_right_3, self.hflip, self.rot)

        if self.transform:
            target_left_1 = self.transform(target_left_1)
            target_right_1 = self.transform(target_right_1)
            inputs_1 = [self.transform(j) for j in inputs_1]

            target_left_2 = self.transform(target_left_2)
            target_right_2 = self.transform(target_right_2)
            inputs_2 = [self.transform(j) for j in inputs_2]

            target_left_3 = self.transform(target_left_3)
            target_right_3 = self.transform(target_right_3)
            inputs_3 = [self.transform(j) for j in inputs_3]

        inputs_1 = torch.cat((torch.unsqueeze(inputs_1[0], 0), torch.unsqueeze(inputs_1[1], 0),
                            torch.unsqueeze(inputs_1[2], 0), torch.unsqueeze(inputs_1[3], 0),
                            torch.unsqueeze(inputs_1[4], 0), torch.unsqueeze(inputs_1[5], 0)))

        inputs_2 = torch.cat((torch.unsqueeze(inputs_2[0], 0), torch.unsqueeze(inputs_2[1], 0),
                            torch.unsqueeze(inputs_2[2], 0), torch.unsqueeze(inputs_2[3], 0),
                            torch.unsqueeze(inputs_2[4], 0), torch.unsqueeze(inputs_2[5], 0)))

        inputs_3 = torch.cat((torch.unsqueeze(inputs_3[0], 0), torch.unsqueeze(inputs_3[1], 0),
                            torch.unsqueeze(inputs_3[2], 0), torch.unsqueeze(inputs_3[3], 0),
                            torch.unsqueeze(inputs_3[4], 0), torch.unsqueeze(inputs_3[5], 0)))

        return inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3

    def __len__(self):
        return len(self.image_filenames)


class TrainSetLoader_all_twoTarget_warploss_2(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TrainSetLoader_all_twoTarget_warploss_2, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale
        self.patch_size = opt.patch_size
        self.data_augmentation = opt.augmentation
        self.hflip = opt.hflip
        self.rot = opt.rot

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):

        inputs_1, inputs_2, target_left_1, target_right_1, target_left_2, target_right_2 = \
            load_image_train_twoTarget_warploss2(self.image_filenames[index])

        if self.patch_size != 0:
            inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3 =\
                get_patch_twoTarget_warploss(inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2,
                                             target_right_2, target_left_3, target_right_3, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3\
                = augment_twoTarget_warploss(inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2,
                                             target_right_2, target_left_3, target_right_3, self.hflip, self.rot)

        if self.transform:
            target_left_1 = self.transform(target_left_1)
            target_right_1 = self.transform(target_right_1)
            inputs_1 = [self.transform(j) for j in inputs_1]

            target_left_2 = self.transform(target_left_2)
            target_right_2 = self.transform(target_right_2)
            inputs_2 = [self.transform(j) for j in inputs_2]

            target_left_3 = self.transform(target_left_3)
            target_right_3 = self.transform(target_right_3)
            inputs_3 = [self.transform(j) for j in inputs_3]

        inputs_1 = torch.cat((torch.unsqueeze(inputs_1[0], 0), torch.unsqueeze(inputs_1[1], 0),
                            torch.unsqueeze(inputs_1[2], 0), torch.unsqueeze(inputs_1[3], 0),
                            torch.unsqueeze(inputs_1[4], 0), torch.unsqueeze(inputs_1[5], 0)))

        inputs_2 = torch.cat((torch.unsqueeze(inputs_2[0], 0), torch.unsqueeze(inputs_2[1], 0),
                            torch.unsqueeze(inputs_2[2], 0), torch.unsqueeze(inputs_2[3], 0),
                            torch.unsqueeze(inputs_2[4], 0), torch.unsqueeze(inputs_2[5], 0)))

        inputs_3 = torch.cat((torch.unsqueeze(inputs_3[0], 0), torch.unsqueeze(inputs_3[1], 0),
                            torch.unsqueeze(inputs_3[2], 0), torch.unsqueeze(inputs_3[3], 0),
                            torch.unsqueeze(inputs_3[4], 0), torch.unsqueeze(inputs_3[5], 0)))

        return inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3

    def __len__(self):
        return len(self.image_filenames)



class TrainSetLoader_all_5f(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TrainSetLoader_all_5f, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale
        self.patch_size = opt.patch_size
        self.data_augmentation = opt.augmentation
        self.hflip = opt.hflip
        self.rot = opt.rot

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):

        inputs, target = load_image_train_all(self.image_filenames[index])

        if self.patch_size != 0:
            inputs, target = get_patch_all(inputs, target, self.patch_size,
                                           self.upscale_factor)

        if self.data_augmentation:
            inputs, target = augment_all(inputs, target, self.hflip, self.rot)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0),
                            torch.unsqueeze(inputs[6], 0), torch.unsqueeze(inputs[7], 0),
                            torch.unsqueeze(inputs[8], 0), torch.unsqueeze(inputs[9], 0)))

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)


class TrainSetLoader_all_7f(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TrainSetLoader_all_7f, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale
        self.patch_size = opt.patch_size
        self.data_augmentation = opt.augmentation
        self.hflip = opt.hflip
        self.rot = opt.rot

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):

        inputs, target = load_image_train_all(self.image_filenames[index])

        if self.patch_size != 0:
            inputs, target = get_patch_all(inputs, target, self.patch_size,
                                           self.upscale_factor)

        if self.data_augmentation:
            inputs, target = augment_all(inputs, target, self.hflip, self.rot)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0),
                            torch.unsqueeze(inputs[6], 0), torch.unsqueeze(inputs[7], 0),
                            torch.unsqueeze(inputs[8], 0), torch.unsqueeze(inputs[9], 0),
                            torch.unsqueeze(inputs[10], 0), torch.unsqueeze(inputs[11], 0),
                            torch.unsqueeze(inputs[12], 0), torch.unsqueeze(inputs[13], 0)
                            ))

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)


class TrainSetLoader_all_single(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TrainSetLoader_all_single, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale
        self.patch_size = opt.patch_size
        self.data_augmentation = opt.augmentation
        self.hflip = opt.hflip
        self.rot = opt.rot

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):

        inputs, target = load_image_train_all(self.image_filenames[index])

        if self.patch_size != 0:
            inputs, target = get_patch_all(inputs, target, self.patch_size,
                                           self.upscale_factor)

        if self.data_augmentation:
            inputs, target = augment_all(inputs, target, self.hflip, self.rot)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0)))

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)


class TestSetLoader(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TestSetLoader, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):
        inputs_lefts, input_right, target = load_image_train_test(self.image_filenames[index], self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            input_right = self.transform(input_right)
            inputs_lefts = [self.transform(j) for j in inputs_lefts]

        inputs_lefts = torch.cat((torch.unsqueeze(inputs_lefts[0], 0), torch.unsqueeze(inputs_lefts[1], 0),
                                  torch.unsqueeze(inputs_lefts[2], 0)))

        return inputs_lefts, input_right, target

    def __len__(self):
        return len(self.image_filenames)


class TestSetLoader_right(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TestSetLoader_right, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):
        input_rights, _, target = load_image_train_test_right(self.image_filenames[index], self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            input_rights = [self.transform(j) for j in input_rights]
            # input_rights = [self.transform(j) for j in input_rights]

        input_rights = torch.cat((torch.unsqueeze(input_rights[0], 0), torch.unsqueeze(input_rights[1], 0),
                                  torch.unsqueeze(input_rights[2], 0)))

        return input_rights, input_rights, target

    def __len__(self):
        return len(self.image_filenames)


class TestSetLoader_all(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TestSetLoader_all, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):
        inputs, target = load_image_train_all_test(self.image_filenames[index], self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0)))

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)


class TestSetLoader_all_5f(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TestSetLoader_all_5f, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):
        inputs, target = load_image_train_all_test(self.image_filenames[index], self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0),
                            torch.unsqueeze(inputs[6], 0), torch.unsqueeze(inputs[7], 0),
                            torch.unsqueeze(inputs[8], 0), torch.unsqueeze(inputs[9], 0)))

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)


class TestSetLoader_all_7f(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TestSetLoader_all_7f, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):
        inputs, target = load_image_train_all_test(self.image_filenames[index], self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0),
                            torch.unsqueeze(inputs[6], 0), torch.unsqueeze(inputs[7], 0),
                            torch.unsqueeze(inputs[8], 0), torch.unsqueeze(inputs[9], 0),
                            torch.unsqueeze(inputs[10], 0), torch.unsqueeze(inputs[11], 0),
                            torch.unsqueeze(inputs[12], 0), torch.unsqueeze(inputs[13], 0)
                            ))

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)


class TestSetLoader_all_twoTar(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TestSetLoader_all_twoTar, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):
        inputs, target_left, target_right = load_image_test_twoTarget(self.image_filenames[index], self.upscale_factor)

        if self.transform:
            target_left = self.transform(target_left)
            target_right = self.transform(target_right)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0)))

        return inputs, target_left, target_right

    def __len__(self):
        return len(self.image_filenames)


class TestSetLoader_singleViewCopy(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TestSetLoader_singleViewCopy, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):
        inputs, target = load_image_train_all_test(self.image_filenames[index], self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[0], 0),
                            torch.unsqueeze(inputs[1], 0), torch.unsqueeze(inputs[2], 0)))

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)


class TestSetLoader_all_crop(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TestSetLoader_all_crop, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):
        inputs, target = load_image_all_test_crop(self.image_filenames[index], self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0)))

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)


class TestSetLoader_all_5f_crop(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TestSetLoader_all_5f_crop, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):
        inputs, target = load_image_all_test_crop(self.image_filenames[index], self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0),
                            torch.unsqueeze(inputs[6], 0), torch.unsqueeze(inputs[7], 0),
                            torch.unsqueeze(inputs[8], 0), torch.unsqueeze(inputs[9], 0)))

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)


class TestSetLoader_all_7f_crop(Dataset):
    def __init__(self, opt, transform=transform()):
        super(TestSetLoader_all_7f_crop, self).__init__()
        self.datafile = opt.datafile
        self.upscale_factor = opt.scale

        groups = [line.rstrip() for line in open(os.path.join(self.datafile))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform

    def __getitem__(self, index):
        inputs, target = load_image_all_test_crop(self.image_filenames[index], self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(j) for j in inputs]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                            torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                            torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0),
                            torch.unsqueeze(inputs[6], 0), torch.unsqueeze(inputs[7], 0),
                            torch.unsqueeze(inputs[8], 0), torch.unsqueeze(inputs[9], 0),
                            torch.unsqueeze(inputs[10], 0), torch.unsqueeze(inputs[11], 0),
                            torch.unsqueeze(inputs[12], 0), torch.unsqueeze(inputs[13], 0)
                            ))

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)

class L1Loss(object):
    def __call__(self, input, target):
        return torch.abs(input - target).mean()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="StereoSR")
    # data loader
    parser.add_argument("--datafile", type=str,
                        default='/gdata1/xurk/StereoSR/data/KITTI2012_video_new/traindata_k12_twotarget_warploss.txt',
                        help="TrainSet path")
    parser.add_argument("--scale", type=int, default=4, help="training batch size")
    parser.add_argument('--augmentation', type=bool, default=True, help='prefix of different dataset')
    parser.add_argument('--patch_size', type=int, default=64, help='file for labels')
    parser.add_argument('--hflip', type=bool, default=True, help='prefix of different dataset')
    parser.add_argument('--rot', type=bool, default=True, help='prefix of different dataset')
    arg = parser.parse_args()
    print(arg.datafile)
    train_set = TrainSetLoader_all_twoTarget_warploss(arg)
    test_set = TestSetLoader_all_twoTar(arg)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=4, shuffle=True)
    # testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
    for i, (inputs_1, inputs_2, inputs_3, target_left_1, target_right_1, target_left_2, target_right_2, target_left_3, target_right_3) in enumerate(training_data_loader):

        if i % 10 == 0:
            print(i)

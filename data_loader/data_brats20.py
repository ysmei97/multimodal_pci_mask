import os
import numpy as np
import scipy.misc

import torch.nn as nn
import torch
from torch.utils.data import Dataset

from src.utils import *


class Brats20DataLoader(Dataset):
    def __init__(self, data_dir, conf='../config/train.conf', train=True):
        label_num = 5
        img_lists = []
        train_config = open(conf).readlines()
        for data in train_config:
            img_lists.append(os.path.join(data_dir, data.strip('\n')))

        print('-------- Loading data from disk --------')
        self.data = []
        self.freq = np.zeros(label_num)
        self.zero_vol = np.zeros((4, 240, 240))
        count = 0
        for subject in img_lists:
            count += 1
            if count % 10 == 0:
                print('loading subject %d' % count)
            volume, label = Brats20DataLoader.get_subject(subject)   # 4 * 155 * 240 * 240,  155 * 240 * 240
            volume = norm_vol(volume)

            self.freq += self.get_freq(label)
            if train is True:
                length = volume.shape[1]
                for i in range(length):
                    name = subject + '=slice' + str(i)
                    if (volume[:, i, :, :] == self.zero_vol).all():  # when training, ignore zero data
                        continue
                    else:
                        self.data.append([volume[:, i, :, :], label[i, :, :], name])
            else:
                volume = np.transpose(volume, (1, 0, 2, 3))
                self.data.append([volume, label, subject])

        self.freq = self.freq / np.sum(self.freq)
        self.weight = np.median(self.freq) / (self.freq + 0.000001)
        print('********  Finish loading data  ********')
        print('********  Weight for all classes  ********')
        print(self.weight)
        if train is True:
            print('********  Total number of 2D images is ' + str(len(self.data)) + ' **********')
        else:
            print('********  Total number of subject is ' + str(len(self.data)) + ' **********')

        print('~' * 50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # ********** get file dir **********
        [image, label, name] = self.data[index]  # get whole data for one subject
        # ********** change data type from numpy to torch.Tensor **********
        image = torch.from_numpy(image).float()  # Float Tensor 4, 240, 240
        label = torch.from_numpy(np.uint8(label)).float()    # Float Tensor 240, 240
        return image, label, name

    @staticmethod
    def get_subject(subject):
        """
        param subject: absolute dir
        return:
        volume  4D numpy    4 * 155 * 240 * 240
        label   4D numpy    155 * 240 * 240
        """
        # **************** get file ****************
        files = os.listdir(subject)
        input_modality_dir = []
        for f in files:
            if f == '.DS_Store':
                continue
            if 'flair.nii' in f:
                path_flair = f
            if 't1.nii' in f:
                path_t1 = f
            if 't1ce.nii' in f:
                path_t1ce = f
            if 't2.nii' in f:
                path_t2 = f
            if 'seg.nii' in f:
                label_dir = f

        input_modality_dir.append(path_flair)
        input_modality_dir.append(path_t1)
        input_modality_dir.append(path_t1ce)
        input_modality_dir.append(path_t2)

        # ********** load 4 modality images **********
        multi_mode_imgs = []  # list size :4      item size: 155 * 240 * 240
        for mod_dir in input_modality_dir:
            path = os.path.join(subject, mod_dir)
            img = load_mha_as_array(path)
            multi_mode_imgs.append(img)

        # ********** get label **********
        label_dir = os.path.join(subject, label_dir)
        label = load_mha_as_array(label_dir)

        volume = np.asarray(multi_mode_imgs)
        return volume, label

    def get_freq(self, label):
        """
        param label: numpy 155 * 240 * 240
        seg classes:
        0 : 'NOT tumor',
        1 : 'NECROTIC/CORE', or NON-ENHANCING tumor CORE
        2 : 'EDEMA',
        4 : 'ENHANCING'
        return:
        """
        class_count = np.zeros(5)
        for i in range(5):
            a = (label == i) + 0
            class_count[i] = np.sum(a)
        # class_count = np.delete(class_count, 3)  # Remove label 3
        # print('class_count: ', class_count)  # CHECK: NO label 3
        return class_count


# test case
if __name__ == "__main__":
    vol_num = 4
    data_dir = './data_sample/'
    conf = '../config/sample_train.conf'
    # test data loader for training data
    brats = Brats20DataLoader(data_dir=data_dir, conf=conf, train=True)
    image2d, label2d, im_name = brats[90]

    print('image size ......')
    print(image2d.shape)             # (4,  240, 240)

    print('label size ......')
    print(label2d.shape)             # (240, 240)
    print(im_name)
    name = im_name.split('/')[-1]
    save_one_image_label(image2d, label2d, 'img_sample/img_label_%s.jpg' %name)

    # test data loader for testing data
    brats_test = Brats20DataLoader(data_dir=data_dir, conf=conf, train=True)
    image_volume, label_volume, subject = brats_test[0]
    print(image_volume.shape)
    print(label_volume.shape)
    print(subject)







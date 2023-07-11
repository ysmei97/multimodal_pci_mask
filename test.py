from src.utils import *

from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys
import torch
import torch.nn as nn
import torch.optim as optim


# ********** Hyper Parameter **********
data_dir = '/home/yongsheng/Downloads/BraTS2020/MICCAI_BraTS2020_TrainingData'
conf_test = 'config/test.conf'
save_dir = 'test_res/'
saved_model_path = 'checkpoint/best_epoch.pth'
batch_size = 1


# multi-GPU
cuda_available = torch.cuda.is_available()
if cuda_available:
    device_ids = [0, 1, 2]  # multi-GPU
    # torch.cuda.set_device(device_ids[0])


def to_var(tensor):
    return Variable(tensor.cuda() if cuda_available else tensor)


def evaluation(net, test_dataset, criterion, A_cat, save_dir=None):
    """
    :param net:
    :param test_dataset:  data loader batch size = 1
    :param criterion:
    :param temporal:
    :return:
    """
    test_loss = []
    iou_class_all = []
    dice_whole_tumor = []
    dice_enhancing_tumor = []
    dice_tumor_core = []
    sen_whole_tumor = []
    sen_enhancing_tumor = []
    sen_tumor_core = []
    pre_whole_tumor = []
    pre_enhancing_tumor = []
    pre_tumor_core = []
    spec_whole_tumor = []
    spec_enhancing_tumor = []
    spec_tumor_core = []
    class_num = 4

    with torch.no_grad():
        net.eval()
        for step, (images_vol, label_vol, subject) in enumerate(test_dataset):
            subj_target = label_vol.long().squeeze()
            subj_predict = torch.zeros(label_vol.squeeze().shape)
            for t in range(155):  #
                image = to_var(images_vol[:, t, ...])
                label = to_var(label_vol[:, t, ...])
                features, predicts = net(image, A_cat)

                loss_valid = criterion(predicts, label.long())
                test_loss.append(float(loss_valid))

                # softmax and reverse
                predicts = one_hot_reverse(predicts)
                subj_predict[t, ...] = predicts.squeeze().long().data

            subj_whole_tumor_dice = dice_wt(subj_predict, subj_target)  # label 1+2+4
            subj_enhancing_tumor_dice = dice_et(subj_predict, subj_target)  # label 4
            subj_tumor_core_dice = dice_tc(subj_predict, subj_target)  # label 1
            subj_whole_tumor_sen = sensitivity_wt(subj_predict, subj_target)  # label 1+2+4
            subj_enhancing_tumor_sen = sensitivity_et(subj_predict, subj_target)  # label 4
            subj_tumor_core_sen = sensitivity_tc(subj_predict, subj_target)  # label 1
            subj_whole_tumor_pre = precision_wt(subj_predict, subj_target)  # label 1+2+4
            subj_enhancing_tumor_pre = precision_et(subj_predict, subj_target)  # label 4
            subj_tumor_core_pre = precision_tc(subj_predict, subj_target)  # label 1
            subj_whole_tumor_spec = specificity_wt(subj_predict, subj_target)  # label 1+2+4
            subj_enhancing_tumor_spec = specificity_et(subj_predict, subj_target)  # label 4
            subj_tumor_core_spec = specificity_tc(subj_predict, subj_target)  # label 1

            dice_whole_tumor.append(subj_whole_tumor_dice)
            dice_enhancing_tumor.append(subj_enhancing_tumor_dice)
            dice_tumor_core.append(subj_tumor_core_dice)
            sen_whole_tumor.append(subj_whole_tumor_sen)
            sen_enhancing_tumor.append(subj_enhancing_tumor_sen)
            sen_tumor_core.append(subj_tumor_core_sen)
            pre_whole_tumor.append(subj_whole_tumor_pre)
            pre_enhancing_tumor.append(subj_enhancing_tumor_pre)
            pre_tumor_core.append(subj_tumor_core_pre)
            spec_whole_tumor.append(subj_whole_tumor_spec)
            spec_enhancing_tumor.append(subj_enhancing_tumor_spec)
            spec_tumor_core.append(subj_tumor_core_spec)

            # save image
            if save_dir is not None:
                hl, name = subject[0].split('/')[-2:]
                img_save_dir = save_dir + hl + '/' + name + '.nii.gz'
                save_array_as_mha(subj_predict, img_save_dir)

        print('Dice for whole tumor is ')
        average_dice_whole_tumor = sum(dice_whole_tumor) / (len(dice_whole_tumor) * 1.0)
        print(average_dice_whole_tumor)
        print('Dice for enhancing tumor is ')
        print(sum(dice_enhancing_tumor) / (len(dice_enhancing_tumor) * 1.0))
        print('Dice for tumor core is ')
        print(sum(dice_tumor_core) / (len(dice_tumor_core) * 1.0))
        print('Sensitivity for whole tumor is ')
        print(sum(sen_whole_tumor) / (len(sen_whole_tumor) * 1.0))
        print('Sensitivity for enhancing tumor is ')
        print(sum(sen_enhancing_tumor) / (len(sen_enhancing_tumor) * 1.0))
        print('Sensitivity for tumor core is ')
        print(sum(sen_tumor_core) / (len(sen_tumor_core) * 1.0))
        print('Precision for whole tumor is ')
        print(sum(pre_whole_tumor) / (len(pre_whole_tumor) * 1.0))
        print('Precision for enhancing tumor is ')
        print(sum(pre_enhancing_tumor) / (len(pre_enhancing_tumor) * 1.0))
        print('Precision for tumor core is ')
        print(sum(pre_tumor_core) / (len(pre_tumor_core) * 1.0))
        print('Specificity for whole tumor is ')
        print(sum(spec_whole_tumor) / (len(spec_whole_tumor) * 1.0))
        print('Specificity for enhancing tumor is ')
        print(sum(spec_enhancing_tumor) / (len(spec_enhancing_tumor) * 1.0))
        print('Specificity for tumor core is ')
        print(sum(spec_tumor_core) / (len(spec_tumor_core) * 1.0))

    return average_dice_whole_tumor, test_loss

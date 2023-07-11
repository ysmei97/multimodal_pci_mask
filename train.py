from src.multi_unet2d import Multi_Unet
from src.utils import *
from src.hscore import *
from data_loader.data_brats20 import Brats20DataLoader
from test import evaluation

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim


# ********** Hyper Parameter **********
train_dir = '/home/yongsheng/Downloads/BraTS2020/MICCAI_BraTS2020_TrainingData'
valid_dir = '/home/yongsheng/Downloads/BraTS2020/MICCAI_BraTS2020_TrainingData'
conf_train = 'config/train.conf'
conf_valid = 'config/valid.conf'
save_dir = 'checkpoint/'

learning_rate = 0.0001
batch_size = 32
epochs = 1000
alpha = 30.0
beta = 12.0
gamma = 2.0

cuda_available = torch.cuda.is_available()
device_ids = [0, 1, 2]       # multi-GPU

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ******************** build model ********************
net = Multi_Unet(1, 5, 32)  # out_lable_class=4, out binary classification one-hot
if cuda_available:
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=device_ids)
    print(' ----- CUDA enabled -----')
else:
    print(' ----- CUDA disabled -----')

# ******************** data preparation  ********************
print('training data ....')
train_data = Brats20DataLoader(data_dir=train_dir, conf=conf_train, train=True)  # 352 objects
print('validation data .....')
valid_data = Brats20DataLoader(data_dir=valid_dir,  conf=conf_valid, train=False)  # 17 objects

# dataloader
train_dataset = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dataset = DataLoader(dataset=valid_data, batch_size=1, shuffle=True, drop_last=True)


def to_var(tensor):
    return Variable(tensor.cuda() if cuda_available else tensor)


def run():
    best_epoch = 0
    score_max = -1.0
    weight = torch.from_numpy(train_data.weight).float()    # weight for all class
    weight = to_var(weight)

    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(weight=None)

    train_hist = []
    corr_hist = []
    corr_12_hist = []
    corr_13_hist = []
    corr_14_hist = []
    corr_23_hist = []
    corr_24_hist = []
    corr_34_hist = []
    total_hist = []

    size = 57600   # h*w*(out_dim*8)
    A1 = torch.rand(int(size), device='cuda') / 2.0
    A2 = torch.rand(int(size), device='cuda') / 2.0
    A3 = torch.rand(int(size), device='cuda') / 2.0
    A4 = torch.rand(int(size), device='cuda') / 2.0
    A5 = torch.rand(int(size), device='cuda') / 2.0
    A6 = torch.rand(int(size), device='cuda') / 2.0
    sum_A = (size * 0.8) * 6.0
    print('A1_init:\n', A1, A1.shape, torch.sum(A1), sum_A)

    for epoch in range(1, epochs + 1):
        print('epoch....................................' + str(epoch))
        train_loss = []
        corr_loss = []
        total_loss = []
        corr_12 = []
        corr_13 = []
        corr_14 = []
        corr_23 = []
        corr_24 = []
        corr_34 = []
        # *************** train model ***************
        print('train ....')
        net.train()
        for step, (image, label, index) in enumerate(train_dataset):
            image = to_var(image)    # 4D tensor   bz * 4(modal) * 240 * 240
            label = to_var(label)    # 3D tensor   bz * 240 * 240 (value 0-4)
            optimizer.zero_grad()

            A_cat = torch.cat(((A1.detach() + A2.detach() + A3.detach()) / 3.0 * 0.8,
                               (A1.detach() + A4.detach() + A5.detach()) / 3.0 * 0.8,
                               (A2.detach() + A4.detach() + A6.detach()) / 3.0 * 0.8,
                               (A3.detach() + A5.detach() + A6.detach()) / 3.0 * 0.8), 0).expand(3, int(4*size))

            features, predicts = net(image, A_cat)

            loss_12 = hscore_A(features[0], features[1], A1.detach())
            loss_13 = hscore_A(features[0], features[2], A2.detach())
            loss_14 = hscore_A(features[0], features[3], A3.detach())
            loss_21 = hscore_A(features[1], features[0], A1.detach())
            loss_23 = hscore_A(features[1], features[2], A4.detach())
            loss_24 = hscore_A(features[1], features[3], A5.detach())
            loss_31 = hscore_A(features[2], features[0], A2.detach())
            loss_32 = hscore_A(features[2], features[1], A4.detach())
            loss_34 = hscore_A(features[2], features[3], A6.detach())
            loss_41 = hscore_A(features[3], features[0], A3.detach())
            loss_42 = hscore_A(features[3], features[1], A5.detach())
            loss_43 = hscore_A(features[3], features[2], A6.detach())

            loss_train = criterion(predicts, label.long())
            train_loss.append(float(loss_train))

            loss_corr = loss_12 + loss_13 + loss_14 + loss_21 + loss_23 + loss_24 + \
                        loss_31 + loss_32 + loss_34 + loss_41 + loss_42 + loss_43

            """Skip the gap in the starting"""
            if epoch >= 5:
                corr_loss.append(float(loss_corr))
                corr_12.append(float(loss_12))
                corr_13.append(float(loss_13))
                corr_14.append(float(loss_14))
                corr_23.append(float(loss_23))
                corr_24.append(float(loss_24))
                corr_34.append(float(loss_34))

            loss = loss_train * alpha + loss_corr / beta
            if epoch >= 5:
                total_loss.append(float(loss))

            loss.backward()
            optimizer.step()

            # ****** save sample image for each epoch ******
            if step % 100 == 0:
                print('..step ....%d' % step)
                print('....loss correlation....%f' % loss_corr)
                print('....loss train....%f' % loss_train)
                print('....loss total....%f' % loss)
                # print('map_size:', map_size)
                predicts = one_hot_reverse(predicts)  # 3D long Tensor  bz * 240 * 240 (val 0-4)
                save_train_images(image, predicts, label, index, epoch, save_dir=save_dir)

        if epoch % 1 == 0:
            A1 = gredient_A(features[0], features[1], A1.detach())
            A2 = gredient_A(features[0], features[2], A2.detach())
            A3 = gredient_A(features[0], features[3], A3.detach())
            A4 = gredient_A(features[1], features[2], A4.detach())
            A5 = gredient_A(features[1], features[3], A5.detach())
            A6 = gredient_A(features[2], features[3], A6.detach())
            A1, A2, A3, A4, A5, A6 = torch.split(projection_A(torch.cat(
                (A1.detach(), A2.detach(), A3.detach(), A4.detach(), A5.detach(), A6.detach()), 0), sum_A, size), size)

        """
        if epoch % 1 == 0:
            # torch.set_printoptions(edgeitems=120)
            print('A1_updated:\n', A1.shape, torch.sum(A1))
            print('A2_updated:\n', A2.shape, torch.sum(A2))
            print('A3_updated:\n', A3.shape, torch.sum(A3))
            print('A4_updated:\n', A4.shape, torch.sum(A4))
            print('A5_updated:\n', A5.shape, torch.sum(A5))
            print('A6_updated:\n', A6.shape, torch.sum(A6))
            # torch.set_printoptions(profile='default')
            # print('feature_map:\n', torch.mean(features[4], 0), torch.mean(features[4], 0).shape)
        
        if epoch % 500 == 0:
            torch.set_printoptions(edgeitems=120)
            print('A1:\n', A1)
            print('A2:\n', A2)
            print('A3:\n', A3)
            print('A4:\n', A4)
            print('A5:\n', A5)
            print('A6:\n', A6)
            torch.set_printoptions(profile='default')
        """

        # ***************** calculate valid loss *****************
        print('valid ....')
        current_score, valid_loss = evaluation(net, valid_dataset, criterion, A_cat, save_dir=None)

        # **************** save loss for one batch ****************
        print('train_epoch_loss ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
        if epoch >= 5:
            print('corr_epoch_loss ' + str(sum(corr_loss) / (len(corr_loss) * 1.0)))
            print('corr_12_loss ' + str(sum(corr_12) / (len(corr_12) * 1.0)))
            print('corr_13_loss ' + str(sum(corr_13) / (len(corr_13) * 1.0)))
            print('corr_14_loss ' + str(sum(corr_14) / (len(corr_14) * 1.0)))
            print('corr_23_loss ' + str(sum(corr_23) / (len(corr_23) * 1.0)))
            print('corr_24_loss ' + str(sum(corr_24) / (len(corr_24) * 1.0)))
            print('corr_34_loss ' + str(sum(corr_34) / (len(corr_34) * 1.0)))
            print('total_epoch_loss ' + str(sum(total_loss) / (len(total_loss) * 1.0)))
        print('valid_epoch_loss ' + str(sum(valid_loss) / (len(valid_loss) * 1.0)))
        train_hist.append(sum(train_loss) / (len(train_loss) * 1.0))
        if epoch >= 5:
            corr_hist.append(sum(corr_loss) / (len(corr_loss) * 1.0))
            corr_12_hist.append(sum(corr_12) / (len(corr_12) * 1.0))
            corr_13_hist.append(sum(corr_13) / (len(corr_13) * 1.0))
            corr_14_hist.append(sum(corr_14) / (len(corr_14) * 1.0))
            corr_23_hist.append(sum(corr_23) / (len(corr_23) * 1.0))
            corr_24_hist.append(sum(corr_24) / (len(corr_24) * 1.0))
            corr_34_hist.append(sum(corr_34) / (len(corr_34) * 1.0)) 
            total_hist.append(sum(total_loss) / (len(total_loss) * 1.0))

        # **************** save model ****************
        if current_score > score_max:
            best_epoch = epoch
            torch.save(net.state_dict(), os.path.join(save_dir, 'best_epoch.pth'))
            score_max = current_score
        print('valid_mean_dice_max ' + str(score_max))
        print('Current Best epoch is %d' % best_epoch)
        if epoch == epochs:
            torch.save(net.state_dict(), os.path.join(save_dir, 'final_epoch.pth'))

    print('Best epoch is %d' % best_epoch)
    print('Training completed!')


if __name__ == '__main__':
    run()


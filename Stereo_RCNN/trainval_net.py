# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick

# Modified by Peiliang Li for Stereo RCNN train
# --------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import argparse
import time
import torch
from torch.autograd import Variable
from model.utils.config import cfg
from model.utils.net_utils import save_checkpoint, clip_gradient
from model.stereo_rcnn.resnet import resnet
from torch.utils.data import Dataset
import cv2
import csv
import json
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Train the Stereo R-CNN network')

    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=500, type=int)  # 12000000

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default='/home/lotte/TTK4900/Stereo_RCNN/models_stereo/models_without_aug/',
                                                                #'/content/gdrive/MyDrive/Stereo_RCNN/models_stereo'
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=8, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)

    # config optimization
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=10, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=6477, type=int)

    args = parser.parse_args()
    return args


class LarvaeDataset(Dataset):
    def __init__(self, file_path, data_aug):
        print('Creating data set...\n')
        self.im_size = (2*1920, 1055)  # Adjust accordingly to own images regarding y-offset.  (width, height)
        print('image size: ', self.im_size)

        self.file_path = file_path

        self.data_aug = data_aug

        self.data_train, self.data_val = self.build_data_set()

        self.complete_dataset = [self.data_train, self.data_val]


        print('Amount of image pairs to train: ', len(self.data_train))
        print('Amount of image pairs for validation: ', len(self.data_val))

    def __len__(self):
        return len(self.complete_dataset)

    def __getitem__(self, index):

        return self.complete_dataset[index]

    def build_data_set(self):
        #data = list()
        check_list = list() # List to check if the video is already added
        train_data = list()
        val_data = list()
        dataset = list()
        with open(self.file_path + 'labels.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for i, row in enumerate(csv_reader):
                #print(row)
                try:
                    if row[1] != '{}' and i > 0:
                        attributes_dict = json.loads(row[1])
                        if attributes_dict['name'] == 'roi':
                            if row[0] not in check_list:

                                print(row[0])

                                img = cv2.imread(self.file_path + row[0])
                                #print('image structure: ', img)

                                # Left and right image in dataset
                                l_img = img[:, :int(img.shape[1] / 2), :]
                                r_img = img[:, int(img.shape[1] / 2) - 1:-1, :]


                                # Left and right RoIs in dataset
                                l_rois = np.array(attributes_dict['left_rois'])
                                r_rois = np.array(attributes_dict['right_rois'])

                                check_list.append(row[0])
                                data_samples = [l_img, r_img, l_rois, r_rois]

                                dataset.append(data_samples)
                except:
                    print('Something went wrong with: ', row[0])



        # split into train and validation sets
        print('Dataset length: ', len(dataset))
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        print('Lenge datasettene: ', train_size, val_size)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        print('Lenge datasettene: ', len(train_dataset), len(val_dataset))

        if self.data_aug:
            for sample_train in train_dataset:
                l_img_train = sample_train[0]
                r_img_train = sample_train[1]
                l_rois_train = sample_train[2]
                r_rois_train = sample_train[3]


                images_1 = []
                images_2 = []

                images_1.append(l_img_train)
                images_2.append(r_img_train)

                # Image augmentation
                augmentation = iaa.Sequential([

                    # 1. Horizontal flip
                    iaa.Fliplr(0.3),

                    # 3. Multiply (the cannels), make the image darker or brighter
                    iaa.Multiply((0.9, 1.1)),  # standard: 0.8, 1.2

                    # 4. Linear contrast, increasing the contrast
                    iaa.LinearContrast((0.8, 1.4)),

                    # Perform methods below only sometimes (70% of images)
                    iaa.Sometimes(0.5,
                                  iaa.GaussianBlur((0, 1.3))#,
                                  #iaa.Rotate((-15, 15)),
                                  )
                ])



                # Prepare bounding boxes for augmentation
                def get_bounding_box(left_bounding_box, right_bounding_box):
                    bbs_left_ = BoundingBoxesOnImage(left_bounding_box, shape=l_img_train.shape)
                    bbs_right_ = BoundingBoxesOnImage(right_bounding_box, shape=r_img_train.shape)
                    return bbs_left_, bbs_right_

                # Prepare bounding boxes for augmentation
                left_bounding_box = []
                right_bounding_box = []
                for l_roi, r_roi in zip(l_rois_train, r_rois_train):
                    left_bb = BoundingBox(x1=l_roi[0], y1=l_roi[1], x2=l_roi[0] + l_roi[2],
                                          y2=l_roi[1] + l_roi[3])
                    right_bb = BoundingBox(x1=r_roi[0], y1=r_roi[1], x2=r_roi[0] + r_roi[2],
                                           y2=r_roi[1] + r_roi[3])
                    left_bounding_box.append(left_bb)
                    right_bounding_box.append(right_bb)

                # Bounding boxes ready for augmentation
                bbs_left, bbs_right = get_bounding_box(left_bounding_box, right_bounding_box)

                t = 1

                while t <= 2:  # 10 images created from 1 image
                    # Equal augmentation on left and right image
                    aug_det = augmentation.to_deterministic()
                    augmented_images_left, aug_bbs_left = aug_det(images=images_1, bounding_boxes=bbs_left)
                    augmented_images_right, aug_bbs_right = aug_det(images=images_2, bounding_boxes=bbs_right)

                    t = t + 1

                    # Convert bounding boxes back to correct format after augmentation
                    l_rois_aug = []
                    r_rois_aug = []
                    for i in range(len(bbs_left.bounding_boxes)):
                        left_after_aug = aug_bbs_left.bounding_boxes[i]
                        right_after_aug = aug_bbs_right.bounding_boxes[i]

                        left_roi = [left_after_aug.x1, left_after_aug.y1, left_after_aug.x2 - left_after_aug.x1,
                                    left_after_aug.y2 - left_after_aug.y1]
                        right_roi = [right_after_aug.x1, right_after_aug.y1,
                                     right_after_aug.x2 - right_after_aug.x1,
                                     right_after_aug.y2 - right_after_aug.y1]

                        l_rois_aug.append(left_roi)
                        r_rois_aug.append(right_roi)

                    l_rois_aug = np.array(l_rois_aug)
                    r_rois_aug = np.array(r_rois_aug)

                    for l_img, r_img in zip(augmented_images_left, augmented_images_right):
                        data_point_train = self.compose_data(l_img, r_img, l_rois_aug, r_rois_aug)
                        train_data.append(data_point_train)

            for sample_val in val_dataset:
                l_img_val = sample_val[0]
                r_img_val = sample_val[1]
                l_rois_val = sample_val[2]
                r_rois_val = sample_val[3]

                data_point_val = self.compose_data(l_img_val, r_img_val, l_rois_val, r_rois_val)
                val_data.append(data_point_val)
            
            return train_data, val_data

        else:
            for sample_train in train_dataset:
                l_img_train = sample_train[0]
                r_img_train = sample_train[1]
                l_rois_train = sample_train[2]
                r_rois_train = sample_train[3]
                data_point_train = self.compose_data(l_img_train, r_img_train, l_rois_train, r_rois_train)
                train_data.append(data_point_train)

            for sample_val in val_dataset:
                l_img_val = sample_val[0]
                r_img_val = sample_val[1]
                l_rois_val = sample_val[2]
                r_rois_val = sample_val[3]

                data_point_val = self.compose_data(l_img_val, r_img_val, l_rois_val, r_rois_val)
                val_data.append(data_point_val)

            return train_data, val_data




    def compose_data(self, l_img, r_img, l_rois, r_rois):
        data_point = list()

        # left and right images
        im_shape = l_img.shape
        im_size_min = np.min(im_shape[0:2])
        im_scale = float(cfg.TRAIN.SCALES[0]) / float(im_size_min)
        l_img = cv2.resize(l_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        r_img = cv2.resize(r_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        l_img = l_img.astype(np.float32, copy=False)
        r_img = r_img.astype(np.float32, copy=False)
        l_img -= cfg.PIXEL_MEANS
        r_img -= cfg.PIXEL_MEANS

        info = np.array([l_img.shape[0], l_img.shape[1], 1.0], dtype=np.float32)

        l_img = np.moveaxis(l_img.copy(), -1, 0)
        r_img = np.moveaxis(r_img.copy(), -1, 0)

        data_point.append(l_img)
        data_point.append(r_img)

        # Image info
        data_point.append(info)

        # left and right ROIS
        l_temp = np.zeros([30, 5])
        l_rois[:, 2] = l_rois[:, 0] + l_rois[:, 2]
        l_rois[:, 3] = l_rois[:, 1] + l_rois[:, 3]
        l_rois = l_rois * im_scale
        l_temp[0:l_rois.shape[0], 0:4] = l_rois
        l_temp[0:l_rois.shape[0], 4] = 1

        r_temp = np.zeros([30, 5])
        r_rois[:, 2] = r_rois[:, 0] + r_rois[:, 2]
        r_rois[:, 3] = r_rois[:, 1] + r_rois[:, 3]
        r_rois = r_rois * im_scale
        r_temp[0:r_rois.shape[0], 0:4] = r_rois
        r_temp[0:r_rois.shape[0], 4] = 1

        data_point.append(l_temp.copy())
        data_point.append(r_temp.copy())

        # Merged ROIS
        merge = np.zeros([30, 5])
        for i in range(30):
            merge[i, 0] = np.min([l_temp[i, 0], r_temp[i, 0]])
            merge[i, 1] = np.min([l_temp[i, 1], r_temp[i, 1]])
            merge[i, 2] = np.max([l_temp[i, 2], r_temp[i, 2]])
            merge[i, 3] = np.max([l_temp[i, 3], r_temp[i, 3]])

        merge[0:r_rois.shape[0], 4] = 1
        data_point.append(merge.copy())

        data_point.append(np.zeros([30, 5]))
        data_point.append(np.zeros([30, 6]))
        data_point.append(r_rois.shape[0])

        return data_point.copy()

    
    
    
def training_process():
    stereoRCNN.train()
    start = time.time()

    im_left_data = torch.FloatTensor(1).cuda()
    im_right_data = torch.FloatTensor(1).cuda()
    im_info = torch.FloatTensor(1).cuda()
    num_boxes = torch.LongTensor(1).cuda()
    gt_boxes_left = torch.FloatTensor([1]).cuda()
    gt_boxes_right = torch.FloatTensor(1).cuda()
    gt_boxes_merge = torch.FloatTensor(1).cuda()
    gt_dim_orien = torch.FloatTensor(1).cuda()
    gt_kpts = torch.FloatTensor(1).cuda()


    for step, data in enumerate(training_loader):
        im_left_data.resize_(data[0].size()).copy_(data[0])
        im_right_data.resize_(data[1].size()).copy_(data[1])
        im_info.resize_(data[2].size()).copy_(data[2])

        gt_boxes_left.resize_(data[3].size()).copy_(data[3])
        gt_boxes_right.resize_(data[4].size()).copy_(data[4])

        gt_boxes_merge.resize_(data[5].size()).copy_(data[5])
        gt_dim_orien.resize_(data[6].size()).copy_(data[6])
        gt_kpts.resize_(data[7].size()).copy_(data[7])
        num_boxes.resize_(data[8].size()).copy_(data[8])

        start = time.time()
        stereoRCNN.zero_grad()
        rois_left, rois_right, cls_prob, bbox_pred, dim_orien_pred, kpts_prob, \
        left_border_prob, right_border_prob, rpn_loss_cls, rpn_loss_box_left_right, \
        RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label = \
            stereoRCNN(im_left_data, im_right_data, im_info, gt_boxes_left, gt_boxes_right,
                       gt_boxes_merge, gt_dim_orien, gt_kpts, num_boxes)

        # Total training loss calculated based on Logarithmic loss and Smooth L1 loss 
        loss = rpn_loss_cls.mean() * torch.exp(-uncert[0]) + uncert[0] + \
               rpn_loss_box_left_right.mean() * torch.exp(-uncert[1]) + uncert[1] + \
               RCNN_loss_cls.mean() * torch.exp(-uncert[2]) + uncert[2] + \
               RCNN_loss_bbox.mean() * torch.exp(-uncert[3]) + uncert[3] + \
               RCNN_loss_dim_orien.mean() * torch.exp(-uncert[4]) + uncert[4] + \
               RCNN_loss_kpts.mean() * torch.exp(-uncert[5]) + uncert[5]

        optimizer.zero_grad()
        loss.backward()
        clip_gradient(stereoRCNN, 10.)
        optimizer.step()

        end = time.time()

        loss_rpn_cls = rpn_loss_cls.item()
        loss_rpn_box_left_right = rpn_loss_box_left_right.item()
        loss_rcnn_cls = RCNN_loss_cls.item()
        loss_rcnn_box = RCNN_loss_bbox.item()
        loss_rcnn_dim_orien = RCNN_loss_dim_orien.item()
        loss_rcnn_kpts = RCNN_loss_kpts
        fg_cnt = torch.sum(rois_label.data.ne(0))
        bg_cnt = rois_label.data.numel() - fg_cnt

        log_string(
            '[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e' % (epoch, step, iters_per_epoch_train, loss.item(), lr))
        log_string('\t\t\tfg/bg=(%d/%d), time cost: %f' % (fg_cnt, bg_cnt, end - start))
        log_string(
           '\t\t\trpn_cls: %.4f, rpn_box_left_right: %.4f, rcnn_cls: %.4f, rcnn_box_left_right %.4f,dim_orien %.4f, kpts %.4f' \
           % (loss_rpn_cls, loss_rpn_box_left_right, loss_rcnn_cls, loss_rcnn_box, loss_rcnn_dim_orien,
              loss_rcnn_kpts))

        del rpn_loss_cls, rpn_loss_box_left_right, RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts

    if epoch % 10 == 0 and epoch > 29:
        save_name = os.path.join(output_dir, 'stereo_rcnn_epoch_{}_loss_{}.pth'.format(epoch, round(loss.item(), 2)))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': stereoRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'uncert': uncert.data,
        }, save_name)

        log_string('save model: {}'.format(save_name))
        end = time.time()
        log_string('time %.4f' % (end - start))


def validation_process():
    #stereoRCNN.eval()

    im_left_data_ = torch.FloatTensor(1).cuda()
    im_right_data_ = torch.FloatTensor(1).cuda()
    im_info_ = torch.FloatTensor(1).cuda()
    num_boxes_ = torch.LongTensor(1).cuda()
    gt_boxes_left_ = torch.FloatTensor([1]).cuda()
    gt_boxes_right_ = torch.FloatTensor(1).cuda()
    gt_boxes_merge_ = torch.FloatTensor(1).cuda()
    gt_dim_orien_ = torch.FloatTensor(1).cuda()
    gt_kpts_ = torch.FloatTensor(1).cuda()

    with torch.no_grad():
        for step_, data_ in enumerate(validation_loader):

            im_left_data_.resize_(data_[0].size()).copy_(data_[0])
            im_right_data_.resize_(data_[1].size()).copy_(data_[1])
            im_info_.resize_(data_[2].size()).copy_(data_[2])


            gt_boxes_left_.resize_(data_[3].size()).copy_(data_[3])
            gt_boxes_right_.resize_(data_[4].size()).copy_(data_[4])

            gt_boxes_merge_.resize_(data_[5].size()).copy_(data_[5])
            gt_dim_orien_.resize_(data_[6].size()).copy_(data_[6])
            gt_kpts_.resize_(data_[7].size()).copy_(data_[7])
            num_boxes_.resize_(data_[8].size()).copy_(data_[8])



            rois_left_, rois_right_, cls_prob_, bbox_pred_, dim_orien_pred_, kpts_prob_, \
            left_border_prob_, right_border_prob_, rpn_loss_cls_, rpn_loss_box_left_right_, \
            RCNN_loss_cls_, RCNN_loss_bbox_, RCNN_loss_dim_orien_, RCNN_loss_kpts_, rois_label_ = \
                stereoRCNN(im_left_data_, im_right_data_, im_info_, gt_boxes_left_, gt_boxes_right_,
                           gt_boxes_merge_, gt_dim_orien_, gt_kpts_, num_boxes_)



            val_loss = rpn_loss_cls_ * torch.exp(-uncert[0]) + uncert[0] + \
                       rpn_loss_box_left_right_ * torch.exp(-uncert[1]) + uncert[1] + \
                       RCNN_loss_cls_ * torch.exp(-uncert[2]) + uncert[2] + \
                       RCNN_loss_bbox_ * torch.exp(-uncert[3]) + uncert[3] + \
                       RCNN_loss_dim_orien_ * torch.exp(-uncert[4]) + uncert[4] + \
                       RCNN_loss_kpts_ * torch.exp(-uncert[5]) + uncert[5]

            log_string_val(
                '[epoch %2d][iter %4d/%4d] validation_loss: %.4f, lr: %.2e' % (
                    epoch, step_, iters_per_epoch_val, val_loss.item(), lr))
            
            del rpn_loss_cls_, rpn_loss_box_left_right_, RCNN_loss_cls_, RCNN_loss_bbox_, RCNN_loss_dim_orien_, RCNN_loss_kpts_




if __name__ == '__main__':
    args = parse_args()

    print('Using config:')
    np.random.seed(cfg.RNG_SEED)

 

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        print('save dir', output_dir)
        os.makedirs(output_dir)
    log_info = open((output_dir + 'train_log.txt'), 'a')
    log_info_val = open((output_dir + 'val_log.txt'), 'a')


    def log_string(out_str):
        log_info.write(out_str + '\n')
        log_info.flush()
        print(out_str)


    def log_string_val(out_str):
        log_info_val.write(out_str + '\n')
        log_info_val.flush()
        print(out_str)
    



    train_dir = '/home/lotte/TTK4900/Stereo_RCNN/data/train_data_stereorcnn/' # Dataset path
              # '/content/gdrive/MyDrive/Stereo_RCNN/data/training_data/'

    dataset = LarvaeDataset(file_path=train_dir, data_aug=False) # True


    dataset_train = dataset[0]
    dataset_val = dataset[1]
    
    train_size = len(dataset_train)
    val_size = len(dataset_val)
    
    print('Batch size: ', args.batch_size)
    training_loader = torch.utils.data.DataLoader(dataset_train, args.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset_val, shuffle=False)
    

    classes = ('__background__', 'Car')

    # initilize the network
    stereoRCNN = resnet(classes, 101, pretrained=True)
    stereoRCNN.create_architecture()


    lr = 0.0001  # Learning rate [0 < lr < 1]

    uncert = Variable(torch.rand(6).cuda(), requires_grad=True)
    torch.nn.init.constant(uncert, -1.0)

    params = []
    for key, value in dict(stereoRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    params += [{'params': [uncert], 'lr': lr}]

    optimizer = torch.optim.Adam(params, lr=lr)
    #optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'stereo_rcnn_epoch_80_loss_-30.17.pth'.format(args.checkepoch,
                                                                                              args.checkpoint))
        print('loading checkpoint %s' % (load_name))
        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']
        stereoRCNN.load_state_dict(checkpoint['model'])
        lr = optimizer.param_groups[0]['lr']
        uncert.data = checkpoint['uncert']
        print('loaded checkpoint %s' % (load_name))

    stereoRCNN.cuda()

    iters_per_epoch_train = int(train_size) / args.batch_size - 1
    iters_per_epoch_val = int(val_size) - 1



    for epoch in range(args.start_epoch, args.max_epochs + 1):
        training_process()
        validation_process()
        print('----------------------------------------------------------------------')


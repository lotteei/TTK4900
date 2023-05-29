from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import torch
import time
import cv2
import sys
import os

from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from model.stereo_rcnn.resnet import resnet
from torch.autograd import Variable
from model.utils.config import cfg
from model.roi_layers import nms


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



sys.path.insert(0, 'home/lotte/TTK4900/Stereo_RCNN/lib/model/')




try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Test the Stereo R-CNN network')

    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models_stereo",
                        type=str)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=6477, type=int)
    args = parser.parse_args()
    return args


args = parse_args()

np.random.seed(cfg.RNG_SEED)

input_dir = args.load_dir + "/"
if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
load_name = os.path.join(input_dir,
                            'stereo_rcnn_epoch_50_loss_-14.99.pth')

kitti_classes = np.asarray(['__background__', 'Car'])



# initilize the network here.
stereoRCNN = resnet(kitti_classes, 101, pretrained=False)
stereoRCNN.create_architecture()
print("load checkpoint %s" % load_name)
checkpoint = torch.load(load_name, map_location = device)
stereoRCNN.load_state_dict(checkpoint['model'])
print('load model successfully!')

eval_thresh = 0.00
vis_thresh = 0.01 # 0.01

stereoRCNN.eval()




def get_pred_boxes(img_left, img_right):

    

    with torch.no_grad():
        # initilize the tensor holder here.
        im_left_data = Variable(torch.FloatTensor(1))
        im_right_data = Variable(torch.FloatTensor(1))
        im_info = Variable(torch.FloatTensor(1))
        num_boxes = Variable(torch.LongTensor(1))
        gt_boxes = Variable(torch.FloatTensor(1))

        



        print("----------------------------------------------------------")
        

        

        # rgb -> bgr
        img_left = img_left.astype(np.float32, copy=False)
        img_right = img_right.astype(np.float32, copy=False)

        img_left -= cfg.PIXEL_MEANS
        img_right -= cfg.PIXEL_MEANS

        im_shape = img_left.shape
        im_size_min = np.min(im_shape[0:2])
        im_scale = float(cfg.TRAIN.SCALES[0]) / float(im_size_min)

        img_left = cv2.resize(img_left, None, None, fx=im_scale, fy=im_scale,
                                interpolation=cv2.INTER_LINEAR)
        img_right = cv2.resize(img_right, None, None, fx=im_scale, fy=im_scale,
                                interpolation=cv2.INTER_LINEAR)

        info = np.array([[img_left.shape[0], img_left.shape[1], im_scale]], dtype=np.float32)

        img_left = torch.from_numpy(img_left)
        img_left = img_left.permute(2, 0, 1).unsqueeze(0).contiguous()

        img_right = torch.from_numpy(img_right)
        img_right = img_right.permute(2, 0, 1).unsqueeze(0).contiguous()

        info = torch.from_numpy(info)

        im_left_data.resize_(img_left.size()).copy_(img_left)
        im_right_data.resize_(img_right.size()).copy_(img_right)
        im_info.resize_(info.size()).copy_(info)

        det_tic = time.time()
        rois_left, rois_right, cls_prob, bbox_pred, bbox_pred_dim, kpts_prob, \
        left_prob, right_prob, rpn_loss_cls, rpn_loss_box_left_right, \
        RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label = \
            stereoRCNN(im_left_data, im_right_data, im_info, gt_boxes, gt_boxes, gt_boxes, gt_boxes,
                        gt_boxes,
                        num_boxes)

        scores = cls_prob.data
        boxes_left = rois_left.data[:, :, 1:5]
        boxes_right = rois_right.data[:, :, 1:5]

        


        bbox_pred = bbox_pred.data
        box_delta_left = bbox_pred.new(bbox_pred.size()[1], 4 * len(kitti_classes)).zero_()
        box_delta_right = bbox_pred.new(bbox_pred.size()[1], 4 * len(kitti_classes)).zero_()

        for keep_inx in range(box_delta_left.size()[0]):
            box_delta_left[keep_inx, 0::4] = bbox_pred[0, keep_inx, 0::6]
            box_delta_left[keep_inx, 1::4] = bbox_pred[0, keep_inx, 1::6]
            box_delta_left[keep_inx, 2::4] = bbox_pred[0, keep_inx, 2::6]
            box_delta_left[keep_inx, 3::4] = bbox_pred[0, keep_inx, 3::6]

            box_delta_right[keep_inx, 0::4] = bbox_pred[0, keep_inx, 4::6]
            box_delta_right[keep_inx, 1::4] = bbox_pred[0, keep_inx, 1::6]
            box_delta_right[keep_inx, 2::4] = bbox_pred[0, keep_inx, 5::6]
            box_delta_right[keep_inx, 3::4] = bbox_pred[0, keep_inx, 3::6]

        box_delta_left = box_delta_left.view(-1, 4)
        box_delta_right = box_delta_right.view(-1, 4)

        box_delta_left = box_delta_left * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        box_delta_right = box_delta_right * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

        box_delta_left = box_delta_left.view(1, -1, 4 * len(kitti_classes))
        box_delta_right = box_delta_right.view(1, -1, 4 * len(kitti_classes))

        pred_boxes_left = bbox_transform_inv(boxes_left, box_delta_left, 1)
        pred_boxes_right = bbox_transform_inv(boxes_right, box_delta_right, 1)
       

        pred_boxes_left = clip_boxes(pred_boxes_left, im_info.data, 1)
        pred_boxes_right = clip_boxes(pred_boxes_right, im_info.data, 1)
       

        pred_boxes_left /= im_info[0, 2].data
        pred_boxes_right /= im_info[0, 2].data
       

        scores = scores.squeeze()[:, 1]
        pred_boxes_left = pred_boxes_left.squeeze()
        pred_boxes_right = pred_boxes_right.squeeze()
        

        det_toc = time.time()
        detect_time = det_toc - det_tic
        print("Detection time: ", detect_time, '\n')



        inds = torch.nonzero(scores > eval_thresh).view(-1)

        if inds.numel() > 0:
            cls_scores = scores[inds]
            _, order = torch.sort(cls_scores, 0, True)


        cls_boxes_left = pred_boxes_left[inds][:, 4:8]
        cls_boxes_right = pred_boxes_right[inds][:, 4:8]
      


        cls_dets_left = torch.cat((cls_boxes_left, cls_scores.unsqueeze(1)), 1)
        cls_dets_right = torch.cat((cls_boxes_right, cls_scores.unsqueeze(1)), 1)

        cls_dets_left = cls_dets_left[order]
        cls_dets_right = cls_dets_right[order]
       
        

        keep = nms(cls_boxes_left[order, :], cls_scores[order], cfg.TEST.NMS)
        keep = keep.view(-1).long()
        cls_dets_left = cls_dets_left[keep]
        cls_dets_right = cls_dets_right[keep]

        l_rois = cls_dets_left.numpy()
        r_rois = cls_dets_right.numpy()

        # color
        red = (0, 0, 128)

        l_pred_boxes = []
        r_pred_boxes = []

        for j, roi in enumerate(l_rois):
            r_score = r_rois[j, -1]
            l_score = l_rois[j, -1]

            if l_score > vis_thresh and r_score > vis_thresh:
                l_bbox_pred = list(int(np.round(x)) for x in l_rois[j, :4])
                r_bbox_pred = list(int(np.round(x)) for x in r_rois[j, :4])

                # Getting on correct format
                l_pred_boxes.append([l_bbox_pred[0], l_bbox_pred[1], l_bbox_pred[2]-l_bbox_pred[0], l_bbox_pred[3]-l_bbox_pred[1]])
                r_pred_boxes.append([r_bbox_pred[0]+ im_shape[1], r_bbox_pred[1], r_bbox_pred[2]-r_bbox_pred[0], r_bbox_pred[3]-r_bbox_pred[1]])


        
        return l_pred_boxes, r_pred_boxes
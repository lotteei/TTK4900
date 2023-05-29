from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import os
import numpy as np
import argparse
import time
import cv2
import torch
from torch.autograd import Variable
from lib.model.utils.config import cfg
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.stereo_rcnn.resnet import resnet
import matplotlib.pyplot as plt
import matplotlib
import stereo_calibration
from stereo_calibration import StereoManager


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Test the Stereo R-CNN network')

    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default='models_stereo/models_without_aug',
                        type=str)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=12, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=6477, type=int)
    args = parser.parse_args()
    return args


def cropped_left_right_image(img):
    img = cv2.imread(img)
    img_shape = img.shape
    
    
   
    img_l = img[:, :int(img_shape[1] / 2), :]
    img_r = img[:, int(img_shape[1] / 2) - 1:-1, :]
    
    '''
    y_offset_value = 25 # Adjust according to the offset of the stereo images # used to be: 75

    left_img_cropped = img_l[y_offset_value:, :, :]
    right_img_cropped = img_r[:-y_offset_value, :, :]
    ''' 



    print('Left image shape: ', img_l.shape)
    print('Right image shape: ', img_r.shape)
    return img_l, img_r


def raw_left_right_image(img):
    img = cv2.imread(img)
    img_shape = img.shape

    left_img_raw = img[:, :int(img_shape[1] / 2), :]
    right_img_raw = img[:, int(img_shape[1] / 2) - 1:-1, :]

    print('left image shape: ', left_img_raw.shape)
    print('Right image shape: ', right_img_raw.shape)

    return left_img_raw, right_img_raw


if __name__ == '__main__':
    
    sm = StereoManager()
    sm.load_calibration('/home/lotte/TTK4900/stereoMaps/stereoMap_padding_4.pickle')
    

    args = parse_args()

    np.random.seed(cfg.RNG_SEED)

    input_dir = args.load_dir + '/'
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'stereo_rcnn_epoch_50_loss_-14.99.pth')  # stereo_rcnn_epoch_105_loss_-35.6236408691406.pth

    kitti_classes = np.asarray(['__background__', 'Car'])

    # initilize the network here.
    stereoRCNN = resnet(kitti_classes, 101, pretrained=False)
    stereoRCNN.create_architecture()

    print('load checkpoint %s' % load_name)
    checkpoint = torch.load(load_name, map_location = device)
    stereoRCNN.load_state_dict(checkpoint['model'])
    print('load model successfully!')

    with torch.no_grad():
        # initilize the tensor holder here.
        im_left_data = Variable(torch.FloatTensor(1, device = device)) #Variable(torch.FloatTensor(1).cuda())
        im_right_data = Variable(torch.FloatTensor(1, device = device))
        im_info = Variable(torch.FloatTensor(1, device = device))
        num_boxes = Variable(torch.FloatTensor(1, device = device))
        gt_boxes = Variable(torch.FloatTensor(1, device = device))

        eval_thresh = 0.00
        vis_thresh = 0.01
        
        stereoRCNN.to(device)
        stereoRCNN.eval()
        
        img_nr = 1
        z_min = 200  # minimum depth to realistic detections
        z_max = 900  # maximum depth to the tank and realistic detections
        image_path = 'data/test_data_3Dpos/*.png'
        path = 'results/3D_pos/3D_pos_detections/'
        with open(path + '3D_pos.txt', 'w') as f:

            for image in glob.glob(image_path):

                
                l_img_cropped, r_img_cropped = cropped_left_right_image(image)

                # rgb -> bgr
                l_img_cropped = l_img_cropped.astype(np.float32, copy=False)
                r_img_cropped = r_img_cropped.astype(np.float32, copy=False)

                l_img_cropped -= cfg.PIXEL_MEANS
                r_img_cropped -= cfg.PIXEL_MEANS

                im_shape = l_img_cropped.shape
                im_size_min = np.min(im_shape[0:2])
                im_scale = float(cfg.TRAIN.SCALES[0]) / float(im_size_min)

                l_img_cropped = cv2.resize(l_img_cropped, None, None, fx=im_scale, fy=im_scale,
                                           interpolation=cv2.INTER_LINEAR)
                r_img_cropped = cv2.resize(r_img_cropped, None, None, fx=im_scale, fy=im_scale,
                                           interpolation=cv2.INTER_LINEAR)

                info = np.array([[l_img_cropped.shape[0], l_img_cropped.shape[1], im_scale]], dtype=np.float32)

                l_img_cropped = torch.from_numpy(l_img_cropped)
                l_img_cropped = l_img_cropped.permute(2, 0, 1).unsqueeze(0).contiguous()

                r_img_cropped = torch.from_numpy(r_img_cropped)
                r_img_cropped = r_img_cropped.permute(2, 0, 1).unsqueeze(0).contiguous()

                info = torch.from_numpy(info)

                im_left_data.resize_(l_img_cropped.size()).copy_(l_img_cropped)
                im_right_data.resize_(r_img_cropped.size()).copy_(r_img_cropped)
                im_info.resize_(info.size()).copy_(info)

                det_tic = time.time()
                rois_left, rois_right, cls_prob, bbox_pred, bbox_pred_dim, kpts_prob, \
                left_prob, right_prob, rpn_loss_cls, rpn_loss_box_left_right, \
                RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label = \
                    stereoRCNN(im_left_data, im_right_data, im_info, gt_boxes, gt_boxes, gt_boxes, gt_boxes, gt_boxes,
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

                box_delta_left = box_delta_left * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS, device = device) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS, device = device)
                box_delta_right = box_delta_right * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS, device = device) \
                                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS, device = device)

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


                l_img_raw, r_img_raw = raw_left_right_image(image)
                im2show_left = np.copy(l_img_raw)
                im2show_right = np.copy(r_img_raw)


                inds = torch.nonzero(scores > eval_thresh).view(-1)

                if inds.numel() > 0:
                    cls_scores = scores[inds]
                    _, order = torch.sort(cls_scores, 0, True)

                det_l = np.zeros([0, 2], dtype=np.int)
                det_r = np.zeros([0, 2], dtype=np.int)
                det_3d = np.zeros([0, 3], dtype=np.int)

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

                l_rois = cls_dets_left.cpu().numpy()
                r_rois = cls_dets_right.cpu().numpy()

                det_number = 1
                stored_x = list()
                stored_y = list()
                stored_z = list()
                stored_dets = list()

                for i, roi in enumerate(l_rois):
                    color = (0,0,255)
                    r_score = r_rois[i, -1]
                    l_score = l_rois[i, -1]
                    if l_score > vis_thresh and r_score > vis_thresh:


                        l_bbox = tuple(int(np.round(x)) for x in l_rois[i, :4])
                        r_bbox = tuple(int(np.round(x)) for x in r_rois[i, :4])

                        # Adjust detected boxes for offset in right image
                        offset_value = 25 #75
                        r_bbox = (r_bbox[0], r_bbox[1] - offset_value, r_bbox[2], r_bbox[3] - offset_value)

                        # Visualize detected boxes
                        im2show_left = cv2.rectangle(im2show_left, l_bbox[0:2], l_bbox[2:4], color, 5)
                        im2show_right = cv2.rectangle(im2show_right, r_bbox[0:2], r_bbox[2:4], color, 5)

                        # Find mid point in left box
                        mid_left = np.array(
                            [l_bbox[0] + int((l_bbox[2] - l_bbox[0]) / 2), l_bbox[1] + int((l_bbox[3] - l_bbox[1]) / 2)],
                            dtype=np.int)
                        # Find mid point in right box
                        mid_right = np.array(
                            [r_bbox[0] + int((r_bbox[2] - r_bbox[0]) / 2), r_bbox[1] + int((r_bbox[3] - r_bbox[1]) / 2)],
                            dtype=np.int)

                        det_l = np.vstack((det_l, mid_left))
                        det_r = np.vstack((det_r, mid_right))



                        im2show_left = cv2.circle(im2show_left, tuple(det_l[i]), 1, color, 10)
                        im2show_right = cv2.circle(im2show_right, tuple(det_r[i]), 1, color, 10)

                        if det_number < 10:

                            im2show_left = cv2.putText(im2show_left, '{:}'.format(det_number),
                                                                   (l_bbox[0]-30, mid_left[1] + 10),
                                                                   cv2.FONT_HERSHEY_DUPLEX,
                                                                   1, (0,0,0), 2, cv2.LINE_AA)

                            im2show_right = cv2.putText(im2show_right, '{:}'.format(det_number),
                                                   (r_bbox[0]-30, mid_right[1] + 10),
                                                   cv2.FONT_HERSHEY_DUPLEX,
                                                   1, (0,0,0), 2, cv2.LINE_AA)

                        else:

                            im2show_left = cv2.putText(im2show_left, '{:}'.format(det_number),
                                                       (l_bbox[0] - 45, mid_left[1] + 10),
                                                       cv2.FONT_HERSHEY_DUPLEX,
                                                       1, (0,0,0), 2, cv2.LINE_AA)

                            im2show_right = cv2.putText(im2show_right, '{:}'.format(det_number),
                                                        (r_bbox[0] - 45, mid_right[1] + 10),
                                                        cv2.FONT_HERSHEY_DUPLEX,
                                                        1, (0,0,0), 2, cv2.LINE_AA)

                        # stored_dets.append(str(det_number))



                        disparity = mid_left[0] - mid_right[0]
                        print('Disparity: ', disparity)

                        sl_key = np.array([mid_left], dtype=np.float32)
                        sr_key = np.array([mid_right], dtype=np.float32)

                        xyz, disparity_points = sm.stereopixel_to_real(sl_key, sr_key)

                        disparity_x = disparity_points[0][0]

                        xyz = xyz[0]
                        # change y-direction upwards
                        xyz[1] = -xyz[1]

                        print('xyz: ', xyz)
                        
                        # Check y coordinate
                        # Check depth coordinate, not added if not valid
                        if z_min < int(xyz[2]) < z_max:    # Hvis dette funker så forandre det  
                            stored_dets.append(str(det_number))
                            stored_x.append(xyz[0])
                            stored_y.append(xyz[1])
                            stored_z.append(int(xyz[2]))

                        img_new = np.hstack((im2show_left, im2show_right))
                        # Resize image
                        im_scale = 0.5
                        img_new = cv2.resize(img_new, None, None, fx=0.3, fy=0.4, interpolation=cv2.INTER_LINEAR)

                        # Save image
                        cv2.imwrite(os.path.join(path, str(img_nr)+ '.jpg'), img_new)


                        cv2.imshow('img', img_new)
                        cv2.waitKey()

                        f.write('\n')
                        f.write('Test nr: ' + str(img_nr))
                        f.write('\n')
                        f.write('Detection nr: ' + str(det_number))
                        f.write('\n')
                        f.write('x: ' + str(xyz[0]))
                        f.write('\n')
                        f.write('y: ' + str(xyz[1]))
                        f.write('\n')
                        f.write('z: ' + str(xyz[2]))
                        f.write('\n')
                        f.write('Disparity: ' + str(disparity_x))
                        f.write('\n')
                        f.write('-----------------------------------------------------------------------------------')
                        det_number += 1

                img_nr += 1

                ################### BAR CHART, DEPTH ##############################

                print('x', stored_x)
                print('y', stored_y)
                print('z', stored_z)


                print('stored detections: ', stored_dets)

                plt.style.use('seaborn-whitegrid')



                x = stored_dets   #stored_z  # ['1', '2', '3', '4', '5', '6']
                y_pos = stored_z

                x_pos = [i+1 for i, _ in enumerate(x)]
    
                plt.figure(figsize=(18, 9))
                plt.bar(x, y_pos, color='green')

                # function to add value labels
                # adapted from https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/
                def addlabels(x, y):
                    for i in range(len(x)):
                        plt.text(i, y[i], y[i], ha='center', fontsize=20)

                # calling the function to add value labels
                addlabels(x_pos, y_pos)

                plt.xlabel('Points of detected objects in image', fontsize=24)
                plt.ylabel('Estimated depth (mm)', fontsize=24)
                plt.title('Estimated depth from stereo images', fontsize=26)

                plt.xticks(size=20)
                plt.yticks(size=20)
                
                
             

                plt.savefig(path+'points_of_detection_'+ str(img_nr-1) + '.png', bbox_inches='tight')
                plt.show()

                ################### plot estimated x/y coordinates ##############################

                # plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
                plt.figure(figsize=(18, 9))
                plt.plot(stored_x, stored_y, 'ro', markersize=12)

                plt.axis([min(stored_x) - 60, max(stored_x) + 60, min(stored_y) - 60, max(stored_y) + 60])


                lab = 1
                # zip joins x and y coordinates in pairs
                for x, y, det in zip(stored_x, stored_y, stored_dets):
                    label = str(lab)  # '{:.2f}'.format(y)

                    plt.annotate(det,(x, y), textcoords='offset points',xytext=(0, 10), ha='center', fontsize=24)
                    lab += 1

                plt.title('Estimated $x/y$ coordinates from stereo images', fontsize=26)
                plt.xlabel('$x$', fontsize=24)
                plt.ylabel('$y$', fontsize=24)

                plt.xticks(size=20)
                plt.yticks(size=20)

                plt.savefig(path + 'x_y_estimates_' + str(img_nr-1) + '.png', bbox_inches='tight')
                plt.show()


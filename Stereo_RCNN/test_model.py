from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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
import csv
import json
from statistics import median
from matplotlib import pyplot as plt

from statistics import mean


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

save_path = 'results/test_model/'

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



# Calculates IoU. Adapted from https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou



if __name__ == '__main__':

    args = parse_args()

    np.random.seed(cfg.RNG_SEED)

    input_dir = args.load_dir + '/'
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'stereo_rcnn_epoch_50_loss_-14.99.pth')

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
        im_left_data = Variable(torch.FloatTensor(1, device = device))
        im_right_data = Variable(torch.FloatTensor(1, device = device))
        im_info = Variable(torch.FloatTensor(1, device = device))
        num_boxes = Variable(torch.FloatTensor(1, device = device))
        gt_boxes = Variable(torch.FloatTensor(1, device = device))
        
        
        eval_thresh = 0.00
        vis_thresh = 0.30 # 0.01

        stereoRCNN.to(device)
        stereoRCNN.eval()


        input_testing_data = 'data/test_data_stereorcnn/'

        with open(input_testing_data + 'labels.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            iou_list = []
            precision_list = []
            recall_list = []
            img_nr = 1

            for i, row in enumerate(csv_reader):
                try:
                    if row[1] != '{}':
                        attributes_dict = json.loads(row[1])
                        if attributes_dict['name'] == 'roi':
                            print('----------------------------------------------------------')
                            print(row[0])

                            img = cv2.imread(input_testing_data + row[0])
                            im_shape = img.shape

                            img_left = img[:, :int(im_shape[1] / 2), :]
                            img_right = img[:, int(im_shape[1] / 2) - 1:-1, :]

                            true_l_rois = np.array(attributes_dict['left_rois'])
                            true_r_rois = np.array(attributes_dict['right_rois'])

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

                            img = cv2.imread(input_testing_data + row[0])
                            im_shape = img.shape

                            img_left = img[:, :int(im_shape[1] / 2), :]
                            img_right = img[:, int(im_shape[1] / 2) - 1:-1, :]



                            im2show_left = np.copy(img_left)
                            im2show_right = np.copy(img_right)


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

                            # colors
                            green = (87, 201, 0)
                            red = (0, 0, 128)

                            l_pred_boxes = []
                            r_pred_boxes = []

                            for j, roi in enumerate(l_rois):
                                color = (np.random.randint(0, 255), np.random.randint(0, 255),
                                         np.random.randint(0, 255))  # (0,0,128)
                                r_score = r_rois[j, -1]
                                l_score = l_rois[j, -1]

                                if l_score > vis_thresh and r_score > vis_thresh:
                                    l_bbox_pred = list(int(np.round(x)) for x in l_rois[j, :4])
                                    r_bbox_pred = list(int(np.round(x)) for x in r_rois[j, :4])

                                    l_pred_boxes.append(l_bbox_pred)
                                    r_pred_boxes.append(r_bbox_pred)


                                # For visualization of detected boxes
                                im2show_left = cv2.rectangle(im2show_left, l_bbox_pred[0:2], l_bbox_pred[2:4], red, 2)
                                im2show_right = cv2.rectangle(im2show_right, r_bbox_pred[0:2], r_bbox_pred[2:4], red, 2)

                            all_detections = len(l_pred_boxes)
                            all_ground_truths = len(true_l_rois)


                            print('Number of detected objects: ', all_detections)
                            print('Number of ground truth boxes: ', all_ground_truths)

                            l_true_boxes = []
                            r_true_boxes = []

                            for k, roi in enumerate(true_l_rois):
                                l_bbox_true = list(int(np.round(x)) for x in true_l_rois[k, :4])
                                r_bbox_true = list(int(np.round(x)) for x in true_r_rois[k, :4])

                                l_bbox_true[2] = l_bbox_true[2] + l_bbox_true[0]
                                l_bbox_true[3] = l_bbox_true[3] + l_bbox_true[1]

                                r_bbox_true[2] = r_bbox_true[2] + r_bbox_true[0]
                                r_bbox_true[3] = r_bbox_true[3] + r_bbox_true[1]


                                l_true_boxes.append(l_bbox_true)
                                r_true_boxes.append(r_bbox_true)


                                # For visualization of ground truth boxes
                                im2show_left = cv2.rectangle(im2show_left, l_bbox_true[0:2], l_bbox_true[2:4], green, 3)
                                im2show_right = cv2.rectangle(im2show_right, r_bbox_true[0:2], r_bbox_true[2:4], green, 3)

                            TP_count = 0
                            for true_BB_left, true_BB_right in zip(l_true_boxes, r_true_boxes):
                                for pred_BB_left, pred_BB_right in zip(l_pred_boxes, r_pred_boxes):


                                    iou_left = bb_intersection_over_union(true_BB_left, pred_BB_left)
                                    iou_right = bb_intersection_over_union(true_BB_right, pred_BB_right)

                                    if iou_left > 0.0 and iou_right > 0.0:
                                        print('iou left', iou_left)
                                        print('iou right', iou_right)
                                        iou_list.append(iou_left)
                                        iou_list.append(iou_right)

                                        im2show_left = cv2.putText(im2show_left, 'IoU: {:.2f}'.format(iou_left),
                                                                   (true_BB_left[0]-50, true_BB_left[1]-8),
                                                                   cv2.FONT_HERSHEY_DUPLEX,
                                                                   1, (0, 0, 0), 1, cv2.LINE_AA)

                                        im2show_right = cv2.putText(im2show_right, 'IoU: {:.2f}'.format(iou_right),
                                                                    (true_BB_right[0]-50, true_BB_right[1] - 8),
                                                                    cv2.FONT_HERSHEY_DUPLEX,
                                                                    1, (0, 0, 0), 1, cv2.LINE_AA)


                                    if iou_left >= 0.4 and iou_right >= 0.4: # Value if categorized as true pos or neg
                                        print('TRUE POSITIVE!')
                                        TP_count += 1




                            print('Number of True Positives in image: ', TP_count)


                            precision = TP_count/all_detections
                            recall = TP_count/all_ground_truths

                            precision_list.append(precision)
                            recall_list.append(recall)


                            print('Precision for this image pair: ', precision)
                            print('Recall for this image pair: ', recall)



                            img = np.hstack((im2show_left, im2show_right))
                            # Resize image
                            im_scale = 0.5
                            img = cv2.resize(img, None, None, fx=0.3, fy=0.4, interpolation=cv2.INTER_LINEAR)
                            
                            #cv2.imshow('image', img)
                            # Save image
                            save_path = 'results/test_model/'
                            cv2.imwrite(os.path.join(save_path, str(img_nr) + '.png'), img)  # img   TODO begge bilder
                            img_nr += 1



                            '''
                            k = cv2.waitKey(0) & 0xFF
                            if k == 27:  # wait for ESC key to exit
                                cv2.destroyAllWindows()

                            elif k == ord('s'):  # wait for 's' key to save and exit
                                val = input('Enter your value: ')
                                print('snap')
                                cv2.imwrite(os.path.join(path, val + '.png'), img)  # img   TODO begge bilder
                                cv2.destroyAllWindows()
                            '''
                                

                except:
                    print('Image: ', row[0], ' not found!')

            print('list of IOU: ', iou_list)





            ################################ IoU distribution ##########################################
            plt.style.use('seaborn-deep')

            bins = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

            plt.hist(iou_list, bins=bins, edgecolor='black')

            median_IoU = median(iou_list)
            print(median_IoU)

            # {:.4f} sier at det skal være 4 desimaler
            plt.axvline(median_IoU, color='#fc4f30', label='Median IoU = ' + '{:.4f}'.format(median_IoU), linewidth=5)

            plt.legend(fontsize=24, loc='upper left')

            # plt.title('IoU of bounding boxes')
            plt.xlabel('Intersection over Union (IoU)', fontsize=24)
            plt.ylabel('Bounding boxes', fontsize=24)

            plt.xticks(size=24)
            plt.yticks(size=24)

            plt.show()


            ################################### Precision and recall #####################################
            print('Precision list: ', precision_list)
            print('Recall list: ', recall_list)

            mean_precision = mean(precision_list)
            mean_recall = mean(recall_list)

            print('mean precision: ', mean_precision)
            print('mean recall: ', mean_recall)

            plt.scatter(recall_list, precision_list, s=800, c='gray', edgecolor='k', linewidth=1,
                        label='Precision/Recall of image samples')

            plt.axhline(mean_precision, c='green', ls='--', label='Precision = ' + '{:.3f}'.format(mean_precision),
                        linewidth=3)
            plt.axvline(mean_recall, c='blue', ls='--', label='Recall = ' + '{:.3f}'.format(mean_recall),
                        linewidth=3)

            plt.legend(fontsize=24, loc='lower left')
            # plt.grid()


            plt.ylim(0.0, 1.1)
            plt.xlim(0.0, 1.1)

            plt.xlabel('Recall', fontsize=24)
            plt.ylabel('Precision', fontsize=24)

            plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], size=24)
            plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], size=24)

            plt.show()


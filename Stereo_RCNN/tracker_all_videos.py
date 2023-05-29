from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import _init_paths
import os
import numpy as np
import argparse
import time
import cv2
import json
import torch
import pickle
from torch.autograd import Variable
from lib.model.utils.config import cfg
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv, kpts_transform_inv, border_transform_inv
from lib.model.stereo_rcnn.resnet import resnet
from tqdm import tqdm
from kalman import kalman
import matplotlib.pyplot as plt
import calibration

#import sort
from sort import *
from stereo_calibration import StereoManager, CameraParameters, StereoParameters

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#create instance of SORT
mot_tracker_left = Sort()
#mot_tracker_right = Sort()

# create instance of stereo manager (calibration)
sm = StereoManager()
sm.load_calibration('stereoMaps/stereoMap_padding_4.pickle')

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    '''
  Parse input arguments
  '''
    parser = argparse.ArgumentParser(description='Test the Stereo R-CNN network')

    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default='models_stereo/models_new',
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
    img_shape = img.shape

    img_l = img[:int(img_shape[0] / 2), :, :]
    img_r = img[int(img_shape[0] / 2) - 1:-1, :, :]

    y_offset_value = 5 # Used to be 75

    left_img_cropped = img_l[y_offset_value:, :, :]
    right_img_cropped = img_r[:-y_offset_value, :, :]

    return left_img_cropped, right_img_cropped


def raw_left_right_image(img):
    img_shape = img.shape
    left_img_raw = img[:int(img_shape[0] / 2), :, :]
    right_img_raw = img[int(img_shape[0] / 2) - 1:-1, :, :]
    return left_img_raw, right_img_raw

def save_to_pickle(save_path, date_name, save_dict, file_name):
    #date_name = dir_path[72:-24].replace('-', '_')
    # Create directory if not exists
    if not os.path.exists(save_path + date_name +'/'):
        os.makedirs(save_path + date_name +'/')

    # Save values to pickle
    with open(save_path + date_name +'/' + date_name + file_name, 'ab') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Getting video files from day
def get_video_files(data_path):
    files_dict = {}
    for dir in os.listdir(data_path):
        folder = os.path.join(data_path, dir)
    
        # checking if it is a file
        if os.path.isdir(folder):
            filelist = glob.glob(os.path.join(folder, '*.mp4'))
            for infile in sorted(filelist):
                files_dict.setdefault(folder, []).append(infile)
    return files_dict


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
        
        stereoRCNN.to(device)

        eval_thresh = 0.00
        vis_thresh = 0.01 # For hoe long the bounding box can be gone before getting new id?
        
        z_depth_min = 200  # Minimum depth of realistic detections
        z_depth_max = 900  # Maximum depth of realistic detections (based on tank depth)

        stereoRCNN.eval()
        
        # Saving path
        path = 'results/tracker/' 
        
        # USB file path
        directory = '/home/lotte/TTK4900/VideoFiles/'

        # Dictionary of filepaths
        files_dict = get_video_files(directory)
        # Deleting paths for videos that are not up for object tracking
        del files_dict['/home/lotte/TTK4900/VideoFiles/2022-06-27_15.36.15']
        
        
        videos = []

        fps = 15                    # Frames per second of video utilized
        save_rate = 15              # The rate of which you want to estimate positions with the Kalman filter
        frame_size = (2*1920, 1080) # (width, height)

        for key in files_dict:
            date_name = key[-19:-6].replace('-', '_')
            print('Key: ', date_name)
            print('Values: ', files_dict[key])
            if date_name == '2022_07_07_06':
                for e in files_dict[key]:
                    videos.append(e)
                    frame_name = date_name +'_'+ e.split('/')[-1][:10]
                    print('N: ', frame_name)
                    
            
        
        # For every folder in files/videos
        for path_key in files_dict:
            print('path_key: ', path_key)
            
            dictionary = {}
            dictionary_velocity_tot = {}
            frame_num_count = 1
            
            
            dictionary_pos = {}
            dictionary_vel = {}
            dictionary_decomp_vel = {}
            dictionary_acc = {}
            
            # For every video in folder
            for video in files_dict[path_key][:21]:
                
                print('Video: ', video, '\n', video[72:-24].replace('-', '_'))
                date_name = video[72:-24].replace('-', '_')
                cap = cv2.VideoCapture(video)
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print('frame_count: ', frame_count)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 

                
                # For video in folder
                for frame_num in tqdm(range(0, frame_count)): # frame_count 900 frames is 1 min

                    print('frame num: ', frame_num_count)

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, image = cap.read()

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
                        stereoRCNN(im_left_data, im_right_data, im_info, gt_boxes, gt_boxes, gt_boxes, \
                        gt_boxes, gt_boxes, num_boxes)

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


                    left_detections = list()
                    right_detections = list()
                    stored_xyz = list()

                    for i, roi in enumerate(l_rois):
                        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))  # (0,0,128)
                        r_score = r_rois[i, -1]
                        l_score = l_rois[i, -1]
                        if l_score > vis_thresh and r_score > vis_thresh:

                            l_bbox = tuple(int(np.round(x)) for x in l_rois[i, :4])
                            r_bbox = tuple(int(np.round(x)) for x in r_rois[i, :4])
                            # Adjust detected boxes for offset in right image
                            offset_value = 25
                            r_bbox = (r_bbox[0], r_bbox[1] - offset_value, r_bbox[2], r_bbox[3] - offset_value)


                            ####### detection list for tracking
                            left_det = list(l_bbox)
                            left_det.append(l_score)
                            right_det = list(r_bbox)
                            right_det.append(r_score)

                            left_detections.append(left_det)
                            right_detections.append(right_det)

                            ########

                            im2show_right = cv2.rectangle(im2show_right, r_bbox[0:2], r_bbox[2:4], color, 5)

                            # Find mid point in left box
                            mid_left = np.array(
                                [l_bbox[0] + int((l_bbox[2] - l_bbox[0]) / 2),
                                l_bbox[1] + int((l_bbox[3] - l_bbox[1]) / 2)],
                                dtype=np.int)
                            # Find mid point in right box
                            mid_right = np.array(
                                [r_bbox[0] + int((r_bbox[2] - r_bbox[0]) / 2),
                                r_bbox[1] + int((r_bbox[3] - r_bbox[1]) / 2)],
                                dtype=np.int)

                            det_l = np.vstack((det_l, mid_left))
                            det_r = np.vstack((det_r, mid_right))



                            im2show_right = cv2.circle(im2show_right, tuple(det_r[i]), 1, color, 10)

                            disparity = mid_left[0] - mid_right[0]
                            #print('Disparity: ', disparity)

                            sl_key = np.array([mid_left], dtype=np.float32)
                            sr_key = np.array([mid_right], dtype=np.float32)

                            xyz, disparity_points = sm.stereopixel_to_real(sl_key, sr_key)

                            disparity_x = disparity_points[0][0]

                            xyz = xyz[0]
                            # change y-direction upwards
                            xyz[1] = -xyz[1]
                            
                            
                            # remove unrealistic detections based of depth of the tank
                            if z_depth_min < int(xyz[2]) < z_depth_max:
                                stored_xyz.append(list(xyz))
                            else:
                                left_detections.remove(left_det)
                                right_detections.remove(right_det)
                                
                            





                    left_detections = np.array(left_detections)
                    right_detections = np.array(right_detections)
                    # track
                    
                    try:
                        track_bbs_ids_left = mot_tracker_left.update(left_detections)
                        
                    except:
                        track_bbs_ids_left = []
                        pass
                            
                        
                        
                    


                    y1_det_left = list()
                    for i in left_detections:
                        y1 = i[1]
                        y1_det_left.append(y1)

                    y1_det_left = np.array(y1_det_left)

                    colors = (255, 0, 0)

                    for i, tracked_left in enumerate(track_bbs_ids_left):
                        tracked_l = tuple(int(np.round(x)) for x in track_bbs_ids_left[i, :5])
                        im2show_left = cv2.rectangle(im2show_left, tracked_l[0:2], tracked_l[2:4], colors, 5)


                        im2show_left = cv2.putText(im2show_left, 'ID: {:}'.format(tracked_l[4]),
                                                (tracked_l[0], tracked_l[1] - 15),
                                                cv2.FONT_HERSHEY_DUPLEX,
                                                1, colors, 1, cv2.LINE_AA)
                        #########
                        y1_tracked = tracked_left[1]

                        diff = np.absolute(y1_det_left - y1_tracked)
                        index = diff.argmin()

                        # xyz of detected object with corresponding ID
                        ID_xyz_item = [tracked_left[4], stored_xyz[index][0], stored_xyz[index][1], stored_xyz[index][2]]

                        if ID_xyz_item[0] in dictionary:
                            dictionary[ID_xyz_item[0]].append(ID_xyz_item[1:4])

                        if not ID_xyz_item[0] in dictionary:
                            dictionary[ID_xyz_item[0]] = []
                            dictionary[ID_xyz_item[0]].append(ID_xyz_item[1:4])



                    if frame_num_count % (save_rate) == 0:
                        dictionary_pos['Frame' + str(frame_num_count)] = {}
                        dictionary_vel['Frame' + str(frame_num_count)] = {}
                        dictionary_decomp_vel['Frame' + str(frame_num_count)] = {}
                        dictionary_acc['Frame' + str(frame_num_count)] = {}
                        
                        for key in dictionary.keys():  
                            
                            if len(dictionary[key]) == (save_rate):
                                pos_list = []
                                decomp_vel_list = []

                                x_0, y_0, z_0, x_1, y_1, z_1, dx_0, dy_0, dz_0, dx_1, dy_1, dz_1 = kalman(dictionary[key], save_rate)
                            
                                pos_list.append(x_1)
                                pos_list.append(y_1)
                                pos_list.append(z_1)

                                decomp_vel_list.append(dx_1)
                                decomp_vel_list.append(dy_1)
                                decomp_vel_list.append(dz_1)

                                
                                # Position to dictionary
                                dictionary_pos['Frame' + str(frame_num_count)][key] = pos_list

                                # Decomposed velocity to dictionary 
                                dictionary_decomp_vel['Frame' + str(frame_num_count)][key] = decomp_vel_list
                                
                                # Velocity to dictionary 
                                velocity = np.sqrt(np.square(dx_1) + np.square(dy_1) + np.square(dz_1))
                                dictionary_vel['Frame' + str(frame_num_count)][key] = velocity

                         
                            


                                if key in dictionary_velocity_tot:
                                    dictionary_velocity_tot[key].append(velocity)
                                    
                                    
                                if not key in dictionary_velocity_tot:
                                    dictionary_velocity_tot[key] = []
                                    dictionary_velocity_tot[key].append(velocity)
                                    
                                # For acceleration purposes
                                acceleration = 0 
                                if len(dictionary_velocity_tot[key]) == 2:
                                    

                                    velocity_0 = dictionary_velocity_tot[key][0]
                                    velocity_1 = dictionary_velocity_tot[key][1]

                                    # acceleration
                                    t = (save_rate/fps)
                                    acceleration = (velocity_1 - velocity_0)/t
                                    dictionary_velocity_tot[key] = [velocity_1]

                                    # Acceleration to dictionary 
                                    dictionary_acc['Frame' + str(frame_num_count)][key] = acceleration

                                

                            else:
                                print('Detected object with ID {} not tracked for the last 15 frames.'.format(key))
                                
                        if frame_num_count > 1:
                            dictionary = {}
                        
                        
                        

                    frame_num_count += 1
                   
                
                    
                cap.release()      
                
            
            save_to_pickle(path, date_name, dictionary_pos, '_3Dpos.pickle')
            save_to_pickle(path, date_name, dictionary_vel, '_vel.pickle')
            save_to_pickle(path, date_name, dictionary_decomp_vel, '_decomp_vel.pickle')
            save_to_pickle(path, date_name, dictionary_acc, '_acc.pickle')
    
    
    
    
    
    



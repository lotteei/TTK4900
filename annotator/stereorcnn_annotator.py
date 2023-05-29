####################################################################
# Developing dataset of stereo images. Adapted from Bjarne Kvæstad # 
####################################################################
import cv2
import numpy as np
import os
import csv
import json
import glob
import random
from tqdm import tqdm
import detection_model
from multiprocessing import Process
from multiprocessing import Queue


# Defining constants
rect_edge = 10
rect_adj = 30
rand_int = random.randint(1, 600) # To get a random image frame from video
frame_width = 1920

data_path = '/home/lotte/TTK4900/VideoFiles/' 
saving_path = '/home/lotte/TTK4900/Stereo_RCNN/data/training_data_stereorcnn/'
label_file = saving_path + 'labels.csv'


mode = 'anno'
ix, iy = -1, -1
l_rois = []
r_rois = []
modify = False
y_offset_value = 25
num_anno_images = 2 # Number of images for annotation 


# roi colors
colors = {0: (0, 128, 0),
          1: (255, 255, 255),
          2: (0, 128, 128),
          3: (0, 0, 0),
          4: (128, 0, 0),
          5: (0, 0, 255),
          6: (0, 0, 128),
          7: (128, 128, 128),
          8: (128, 0, 128),
          9: (0, 255, 255),
          10: (0, 255, 0),
          11: (255, 0, 255),
          12: (255, 255, 0),
          13: (255, 0, 0),
          14: (128, 128, 0),
          15: (192, 192, 192)}

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

videos = get_video_files(data_path)

def get_bbox_rois(queue):
    print('Stereo model prediction: Running...')

    for key in videos:
        for video in videos[key]:

            ### Setting values for each video ###
            rect_edge = 10
            rect_adj = 30
            ######################################



            cap = cv2.VideoCapture(video)  # reading video file
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


            rect_edge = int(rect_edge * frame_width / 3264)
            rect_adj = int(rect_adj * frame_width / 3264)

        

            for frame_num in tqdm(range(rand_int, int(frame_count), int(frame_count / num_anno_images))): # Adjust the number '30' to annotate more/less images 
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, img = cap.read()
                im_shape = img.shape
                
                l_img = img[:int(im_shape[0] / 2), :, :]
                r_img = img[int(im_shape[0] / 2) - 1:-1, :, :]

                l_img = l_img[y_offset_value:, :, :]
                r_img = r_img[:-y_offset_value, :, :]

                ### PUT frame_num, [l_rois, r_rois] from image with stereomodel ### 
                queue.put([frame_num, detection_model.get_pred_boxes(l_img, r_img)])
            
        


            queue.put([None, [None, None]])
    print('Stereo model prediction: Done')


def annotate(queue):
    print('Annotator: Running...')
    global mode 

    # mouse callback function
    def mark(event, x, y, flags, param):
        global ix, iy, mode, modify, roi_id, roi_side

        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = x, y
            modify = True
            for i in range(len(l_rois)):
                p2 = (l_rois[i][0] + l_rois[i][2], l_rois[i][1] + l_rois[i][3])

                if p2[0] - rect_adj < x < p2[0] + rect_adj and p2[1] - rect_adj < y < p2[1] + rect_adj:
                    if mode == 'anno':
                        roi_id = i
                        mode = 'adj'
                    return

            for i in range(len(l_rois)):
                l_p1 = (l_rois[i][0], l_rois[i][1])
                l_p2 = (l_rois[i][0] + l_rois[i][2], l_rois[i][1] + l_rois[i][3])
                r_p1 = (r_rois[i][0], r_rois[i][1])
                r_p2 = (r_rois[i][0] + r_rois[i][2], r_rois[i][1] + r_rois[i][3])

                if l_p1[0] < x < l_p2[0] and l_p1[1] < y < l_p2[1]:
                    if mode == 'del':
                        l_rois.pop(i)
                        r_rois.pop(i)
                        modify = False
                    elif mode == 'anno':
                        roi_id = i
                        roi_side = 'l'
                        ix = x - l_rois[roi_id][0]
                        iy = y - l_rois[roi_id][1]
                        mode = 'move'
                    return

                if r_p1[0] < x < r_p2[0] and r_p1[1] < y < r_p2[1]:
                    if mode == 'del':
                        l_rois.pop(i)
                        r_rois.pop(i)
                        modify = False
                    elif mode == 'anno':
                        roi_id = i
                        roi_side = 'r'
                        ix = x - r_rois[roi_id][0]
                        iy = y - r_rois[roi_id][1]
                        mode = 'move'
                    return

            if mode != 'del' and x < frame_width:
                l_rois.append([ix, iy, 0, 0])
                r_rois.append([frame_width + ix, iy, 0, 0])
            else:
                modify = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if modify:
                if mode == 'anno' and x < frame_width:
                    start_x, start_y = ix, iy
                    end_x, end_y = x, y
                    l_rois[-1][2] = end_x - start_x  # width
                    l_rois[-1][3] = end_y - start_y  # height

                    start_x, start_y = r_rois[-1][0], r_rois[-1][1]
                    end_x, end_y = frame_width + x, y
                    r_rois[-1][2] = end_x - start_x  # width
                    r_rois[-1][3] = end_y - start_y  # height
                elif mode == 'move':
                    if roi_side == 'l':
                        l_rois[roi_id][0] = x - ix
                        l_rois[roi_id][1] = y - iy
                        r_rois[roi_id][1] = y - iy
                    elif roi_side == 'r':
                        r_rois[roi_id][0] = x - ix
                        r_rois[roi_id][1] = y - iy
                        l_rois[roi_id][1] = y - iy
                elif mode == 'adj' and x < frame_width:
                    w = x - l_rois[roi_id][0]
                    h = y - l_rois[roi_id][1]
                    if w < 0:
                        w = 0
                    if h < 0:
                        h = 0

                    l_rois[roi_id][2] = w
                    l_rois[roi_id][3] = h

                    r_rois[roi_id][2] = w
                    r_rois[roi_id][3] = h

        elif event == cv2.EVENT_LBUTTONUP:
            if modify:
                modify = False
                if mode == 'move' or mode == 'adj':
                    mode = 'anno'
                    return
                if mode == 'anno':
                    # roi
                    l_start_x, l_start_y = l_rois[-1][0], l_rois[-1][1]
                    l_end_x, l_end_y = x, y

                    if l_end_x > l_start_x:
                        l_rois[-1][0] = l_start_x
                        l_rois[-1][2] = l_end_x - l_start_x  # width
                    else:
                        l_rois[-1][0] = l_end_x
                        l_rois[-1][2] = l_start_x - l_end_x  # width

                    if l_end_y > l_start_y:
                        l_rois[-1][1] = l_start_y
                        l_rois[-1][3] = l_end_y - l_start_y  # height
                    else:
                        l_rois[-1][1] = l_end_y
                        l_rois[-1][3] = l_start_y - l_end_y  # height

                    r_start_x, r_start_y = r_rois[-1][0], r_rois[-1][1]
                    r_end_x, r_end_y = frame_width + x, y

                    if r_end_x > r_start_x:
                        r_rois[-1][0] = r_start_x
                        r_rois[-1][2] = r_end_x - r_start_x  # width
                    else:
                        r_rois[-1][0] = r_end_x
                        r_rois[-1][2] = r_start_x - r_end_x  # width

                    if r_end_y > r_start_y:
                        r_rois[-1][1] = r_start_y
                        r_rois[-1][3] = r_end_y - r_start_y  # height
                    else:
                        r_rois[-1][1] = r_end_y
                        r_rois[-1][3] = r_start_y - r_end_y  # height

                    if l_rois[-1][2] < 10 or l_rois[-1][3] < 10:
                        l_rois.pop(-1)
                        r_rois.pop(-1)

    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', mark)

    
    for key in videos:
        date_name = key[-19:-6].replace("-", "_")
        for video in videos[key]:
          

            ### Setting values for each video ###
            l_rois = []
            r_rois=[]
            mode = 'anno'
            rect_edge = 10
            rect_adj = 30
            ######################################



            cap = cv2.VideoCapture(video)  # reading video file
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            rect_edge = int(rect_edge * frame_width / 3264)
            rect_adj = int(rect_adj * frame_width / 3264)

        

            # Adjust the number '30' to annotate more/less images 
            while True:
                frame_num, [l_rois, r_rois] = queue.get()
                frame_name = date_name +'_'+ video.split('/')[-1][:10] + "_" + str(frame_num) + '.png'
               
                # Check if queue is empty
                if frame_num is None:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, img = cap.read()
                im_shape = img.shape
                
                l_img = img[:int(im_shape[0] / 2), :, :]
                r_img = img[int(im_shape[0] / 2) - 1:-1, :, :]

                l_img = l_img[y_offset_value:, :, :]
                r_img = r_img[:-y_offset_value, :, :]

                
                
                img = np.hstack((l_img, r_img))
                

                while (1):
                    vis_img = img.copy()
                    for i, roi in enumerate(l_rois):
                        if i > 15:
                            i = 15
                        p1 = (roi[0], roi[1])
                        p2 = (roi[0] + roi[2], roi[1] + roi[3])
                        cv2.rectangle(vis_img, p1, p2, colors[i], rect_edge)
                        cv2.rectangle(vis_img, (p2[0] - rect_adj, p2[1] - rect_adj), (p2[0] + rect_adj, p2[1] + rect_adj),
                                    colors[i], rect_edge)

                    for i, roi in enumerate(r_rois):
                        if i > 15:
                            i = 15
                        p1 = (roi[0], roi[1])
                        p2 = (roi[0] + roi[2], roi[1] + roi[3])
                        cv2.rectangle(vis_img, p1, p2, colors[i], rect_edge)

                    
                    #frame_name = video.split('/')[-1][:-4] + "_" + str(frame_num) + ".png"
                    cv2.imshow('image', vis_img)
                    k = cv2.waitKey(1) & 0xFF

                    if k == ord('d'):
                        mode = 'del'
                    if k == ord('a'):
                        mode = 'anno'
                    if k == ord('w'):
                        print("Skipped: ", frame_name)
                        l_rois = []
                        r_rois = []
                        break
                    if k == ord('s'):
                        if len(l_rois) == 0:
                            break

                        for r_roi in r_rois:
                            r_roi[0] = r_roi[0] - frame_width

                        # Create CSV
                        region_attributes = {"name": "roi",
                                                "left_rois": l_rois,
                                                "right_rois": r_rois}
                        

                        
                        region_attributes = json.dumps(region_attributes)

                        

                        cv2.imwrite(saving_path + frame_name, img)
                        size = os.path.getsize(saving_path + frame_name)



                        with open(label_file, 'a+') as f:
                            writer = csv.writer(f)
                            writer.writerow([frame_name, region_attributes])
                            

                        print("Saved: ", frame_name, "\nRois: ", l_rois, '\n', r_rois)
                        l_rois = []
                        r_rois = []
                        break
                    elif k == 27:
                        cv2.destroyAllWindows()
                        
    print('Annotator: Done')







if __name__ == '__main__':
 
    # Create the shared queue
    queue = Queue()

    # Start the annotating
    annotate_process = Process(target=annotate, args=(queue,))
    annotate_process.start()

    # Start the stereo prediction of images
    get_bbox_process = Process(target=get_bbox_rois, args=(queue,))
    get_bbox_process.start()
    
    # Wait for all processes to finish
    get_bbox_process.join()
    annotate_process.join()

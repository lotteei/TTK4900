import calibration

from torch.autograd import Variable
from scipy.signal import fftconvolve
from lib.model.utils.config import cfg
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv, kpts_transform_inv, border_transform_inv
from lib.model.stereo_rcnn.resnet import resnet

import matplotlib.pyplot as plt

import _init_paths
import os
import numpy as np
import argparse
import time
import cv2
import torch

# TODO test images
img_l_path = 'demo/chess2_left.png'  
img_r_path = 'demo/chess2_right.png'  

img_left = cv2.imread(img_l_path)
img_right = cv2.imread(img_r_path)



################## CALIBRATION #########################################################

img_right, img_left = calibration.undistortRectify(img_right, img_left)

########################################################################################

########## Offset #################################



def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out


# gray-scale image used in cross-correlation
right_img = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
left_img = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# cross-correlation
corr = normxcorr2(right_img, left_img)
# cross corr between equal image
corr_1 = normxcorr2(right_img, right_img)

# find match
y, x = np.unravel_index(np.argmax(corr), corr.shape)
# peak left image
y_1, x_1 = np.unravel_index(np.argmax(corr_1), corr_1.shape)
# offset
y_offset_value = y_1 - y

print("\nOffset value: ", y_offset_value)


#####################################################################################


height, width, channels = img_left.shape

print(height,width)


### RESIZE ###########
#img_left = cv2.resize(img_left, (640, 360))
#img_right = cv2.resize(img_right, (640, 360))
#######################


#### LINJE #####
start_point_1 = (0, 115)
end_point_1 = (780, 115)

start_point_2 = (0, 264)
end_point_2 = (780, 264)


# Green color in BGR
color = (0, 0, 255)

# Line thickness of 9 px
thickness = 2


image_L = cv2.resize(img_left, (544, 308)) # 3264x1848
image_R = cv2.resize(img_right, (544, 308)) # 3264x1848

# Using cv2.line() method
# Draw a diagonal green line with thickness of 9 px
image_L = cv2.line(image_L, start_point_1, end_point_1, color, thickness)
image_R = cv2.line(image_R, start_point_1, end_point_1, color, thickness)

image_L = cv2.line(image_L, start_point_2, end_point_2, color, thickness)
image_R = cv2.line(image_R, start_point_2, end_point_2, color, thickness)
##########################




############ vis images ################################
img = np.hstack((image_L, image_R))




# Save image
path = 'results/rectified_img/'
cv2.imwrite(os.path.join(path, 'rectified_image.jpg'), img)

cv2.imshow("img", img)
cv2.waitKey()
###########################################################





############ vis images plot ###################################

# convert color image into grayscale image
img1 = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)
img2 = cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY)

# Draw the rectified images
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].imshow(img1,  cmap='gray')
axes[1].imshow(img2,  cmap='gray')
axes[0].axhline(410)
axes[1].axhline(410)

axes[0].axhline(688)
axes[1].axhline(688)

axes[0].axhline(925)
axes[1].axhline(925)




plt.suptitle("Rectified images", fontsize=22)
plt.savefig(path + "rectified_images.png")
plt.show()

cv2.waitKey(1)

########################################################################




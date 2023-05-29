import cv2

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap_padding_4.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


def undistortRectify(frameR, frameL):
    undistortedL = cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_CUBIC)
    undistortedR = cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_CUBIC)

    return undistortedR, undistortedL


import os
from turtle import right
import cv2
import pickle
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
import yaml
from threading import Thread, Lock
from queue import Queue
import calibration

#-----------------------------------------------------------------------
# Code adapted from https://github.com/bjarnekvae/Stereo-RCNN/tree/generalize
#-----------------------------------------------------------------------


def save_calibration_data(file_path, calibration_data):
    for key in calibration_data.keys():
        calibration_data[key] = calibration_data[key].tolist()
    with open(file_path, 'w') as cal_file:
        cal_file.write(yaml.dump(calibration_data))


def load_calibration_data(file_path):
    with open(file_path, 'r') as cal_file:
        calibration_data = yaml.load(cal_file, Loader=yaml.FullLoader)
    for key in calibration_data.keys():
        calibration_data[key] = np.asarray(calibration_data[key])
    return calibration_data


def get_files_from_dir(dir_path):
    return [f for f in listdir(dir_path) if isfile(join(dir_path, f))]


def umeyama(a, b):
    """
    Calculates the transformation between points a and points b.
    a and b must have the same shape where rows are 3 (xyz) and
    cols are the amount of points n.

    Returns the transformation matrix T so that:
    |x_b|           |x_a|
    |y_b|  =  T  *  |y_b|
    |z_b|           |z_b|
    | 1 |           | 1 |

    """
    d = a.shape[0]
    n = a.shape[1]

    a_c = np.mean(a, axis=1)
    b_c = np.mean(b, axis=1)

    aa = a - np.tile(a_c, [n, 1]).T
    bb = b - np.tile(b_c, [n, 1]).T
    test = aa @ bb.T

    u, s, vh = np.linalg.svd(test)
    v = vh.T

    R = v @ np.diag([1, 1, np.linalg.det(v @ u.T)]) @ u.T
    t = b_c - R @ a_c

    T = np.diag([0, 0, 0, 1]).astype(np.float)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T

class CameraParameters:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.r = None
        self.p = None
        self.map1 = None
        self.map2 = None

    def set_calibration_parameters(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def set_rectify_parameters(self, r, p):
        self.r = r
        self.p = p

    def set_rectify_maps(self, map1, map2):
        self.map1 = map1
        self.map2 = map2


class StereoParameters:
    def __init__(self):
        self.r = None
        self.t = None
        self.q = None

    def set_calibration_parameters(self, r, t):
        self.r = r
        self.t = t

    def set_rectify_parameters(self, q):
        self.q = q


class StereoManager:
    def __init__(self, calibration_file: str = None):
        self.cam_parameters = [CameraParameters(), CameraParameters()] # Cam paramters [left cam, right cam]
        self.stereo_parameters = StereoParameters()
        self.image_size = None

        self.disparity_method = None
        self.nThread_Lock = Lock()
        self.chess_q = Queue(100)
        self.img_q = Queue(100)


    def _find_chess_handler(self, images_left, images_right, board_params: tuple):

        for i, fname in enumerate(images_right):
            self.img_q.put((images_left[i], images_right[i]))


        for i in range(0, 10):
            Thread(target=self._find_chess, args=(board_params,), daemon=True).start()



    def _find_chess(self, board_params: tuple):
        while not self.img_q.empty():
            (image_left, image_right) = self.img_q.get()

            img_l = cv2.imread(image_left)
            img_r = cv2.imread(image_right)

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (board_params[0], board_params[1]), None)
            if ret_l:
                ret_r, corners_r = cv2.findChessboardCorners(gray_r, (board_params[0], board_params[1]), None)
            else:
                ret_r = False
                corners_r = corners_l

            self.chess_q.put((gray_l, ret_l, corners_l, gray_r, ret_r, corners_r))

    def calibrate_from_dir(self, dir_path: str, board_params: tuple):

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC


        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-5)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_CHECK_COND

        objp = np.zeros((1, board_params[0] * board_params[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:board_params[0], 0:board_params[1]].T.reshape(-1, 2) * board_params[2]
        print(f"Object shape: {objp.shape}")

        objpoints = []  # 3d point in real world space
        imgpoints_l = []  # 2d points in image plane.
        imgpoints_r = []  # 2d points in image plane.

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        images_left = [dir_path + o for o in os.listdir(dir_path) if o.startswith('l_') and o.endswith('.png')]
        images_right = [dir_path + o for o in os.listdir(dir_path) if o.startswith('r_') and o.endswith('.png')]
        

        # TODO
        print("Amount images_left: ", len(images_left))
        print("Amount images_right: ", len(images_right))

        images_left.sort()
        images_right.sort()

       
        self.image_size = cv2.imread(images_left[0], cv2.IMREAD_GRAYSCALE).shape[::-1]

        print("Image size: ", self.image_size)

        window_downscale = 3
        
        window_size = (int(self.image_size[0] * 2 / window_downscale), int(self.image_size[1] / window_downscale))
        cv2.namedWindow('Match', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Match', window_size)

       
        chess_thr = Thread(target=self._find_chess_handler, args=(images_left, images_right, board_params,), daemon=True)
        chess_thr.start()
        chess_thr.join()

        print("Stereo manager: Detecting chessboard")
        for i in range(len(images_right)):
            print(f"     Processing image pair {i + 1} of {len(images_right)}", end="")


            (gray_l, ret_l, corners_l, gray_r, ret_r, corners_r) = self.chess_q.get()
            
            if ret_l and ret_r:
                # If found, add object points, image points (after refining them)

                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

                gray_l = cv2.cvtColor(gray_l, cv2.COLOR_GRAY2BGR)
                gray_r = cv2.cvtColor(gray_r, cv2.COLOR_GRAY2BGR)
                vis_l = cv2.drawChessboardCorners(gray_l, (board_params[0], board_params[1]), corners_l, ret_l)
                vis_r = cv2.drawChessboardCorners(gray_r, (board_params[0], board_params[1]), corners_r, ret_r)


                print(" - board found, Accept? [y/n] ")
                

                cv2.imshow("Match", np.concatenate((vis_l, vis_r), axis=1))
                keyCode = cv2.waitKey() & 0xFF
                # Stop the program on the ESC key

                if keyCode == ord("y"):
                    imgpoints_l.append(corners_l)
                    imgpoints_r.append(corners_r)
                    objpoints.append(objp)
                    print("     Accepted")
                elif keyCode == ord("n"):
                    print("     Rejected")
                    pass

            else:
                cv2.imshow("Match", np.concatenate((gray_l, gray_r), axis=1))
                cv2.waitKey(1000)
                print(" - board not found")
        cv2.destroyAllWindows()

        print("Stereo manager: Calibrating camera")

        ############## CALIBRATION #######################################################

        print(f"1 argument: {len(objpoints)}, 2 arguement {len(imgpoints_l)}, flags: {calibration_flags}")
        rt1, m1, d1, r1, t1 = cv2.fisheye.calibrate(objpoints, imgpoints_l, self.image_size, None, None, None, None,
                                                    flags=calibration_flags)
        rt2, m2, d2, r2, t2 = cv2.fisheye.calibrate(objpoints, imgpoints_r, self.image_size, None, None, None, None,
                                                    flags=calibration_flags)

        print('m1: ', m1)
        print('m2: ', m2)

        # Calculate reprojection error:
        l_mean_error = 0
        for i in range(len(objpoints)):
            imgpoints_l_reproj, _ = cv2.fisheye.projectPoints(objpoints[i], r1[i], t1[i], m1, d1)
            imgpoints_l_reproj = np.moveaxis(imgpoints_l_reproj, 0, 1)
            error = cv2.norm(imgpoints_l[i], imgpoints_l_reproj, cv2.NORM_L2) / len(imgpoints_l_reproj)
            l_mean_error += error

        r_mean_error = 0
        for i in range(len(objpoints)):
            imgpoints_r_reproj, _ = cv2.fisheye.projectPoints(objpoints[i], r2[i], t2[i], m2, d2)
            imgpoints_r_reproj = np.moveaxis(imgpoints_r_reproj, 0, 1)
            error = cv2.norm(imgpoints_r[i], imgpoints_r_reproj, cv2.NORM_L2) / len(imgpoints_r_reproj)
            r_mean_error += error

        print(f"    Reprojection error [px]: "
              f"Left cam: {l_mean_error / len(objpoints):.4f}    "
              f"Right cam: {r_mean_error / len(objpoints):.4f}")

        if rt1 and rt2:
            print("Stereo manager: Calibrating stereo")
            self.cam_parameters[0].set_calibration_parameters(m1, d1)
            self.cam_parameters[1].set_calibration_parameters(m2, d2)

            for i in range(0, len(imgpoints_l)):
                imgpoints_l[i] = np.moveaxis(imgpoints_l[i], 0, 1)
                imgpoints_r[i] = np.moveaxis(imgpoints_r[i], 0, 1)

            ########## Stereo Vision Calibration #############################################
            ret, m1, d1, m2, d2, r, t = cv2.fisheye.stereoCalibrate(objpoints,
                                                                  imgpoints_l,
                                                                  imgpoints_r,
                                                                  self.cam_parameters[0].camera_matrix,
                                                                  self.cam_parameters[0].dist_coeffs,
                                                                  self.cam_parameters[1].camera_matrix,
                                                                  self.cam_parameters[1].dist_coeffs,
                                                                  self.image_size,
                                                                  flags=flags) #flags

            if ret:
                print("Stereo manager: Calibrating image rectification")
                self.stereo_parameters.set_calibration_parameters(r, t)
            ########## Stereo Rectification #################################################
                r1, r2, p1, p2, q = cv2.fisheye.stereoRectify(self.cam_parameters[0].camera_matrix,
                                                                      self.cam_parameters[0].dist_coeffs,
                                                                      self.cam_parameters[1].camera_matrix,
                                                                      self.cam_parameters[1].dist_coeffs,
                                                                      self.image_size,
                                                                      self.stereo_parameters.r,
                                                                      self.stereo_parameters.t,
                                                                      flags=0)
                self.cam_parameters[0].set_rectify_parameters(r1, p1)
                self.cam_parameters[1].set_rectify_parameters(r2, p2)
                self.stereo_parameters.set_rectify_parameters(q)

                for i in range(len(self.cam_parameters)):
                    map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.cam_parameters[i].camera_matrix,
                                                             self.cam_parameters[i].dist_coeffs,
                                                             self.cam_parameters[i].r,
                                                             self.cam_parameters[i].p,
                                                             self.image_size,
                                                             cv2.CV_32FC1)
                    self.cam_parameters[i].set_rectify_maps(map1, map2)
                print("Stereo manager: Calibration finished")



                ############## SAVE CALIBRATION ###############################################
                print("Saving parameters!")
                cv_file = cv2.FileStorage('stereoMap_padding_4.xml', cv2.FILE_STORAGE_WRITE)

                cv_file.write('stereoMapL_x', self.cam_parameters[0].map1)
                cv_file.write('stereoMapL_y', self.cam_parameters[0].map2)
                cv_file.write('stereoMapR_x', self.cam_parameters[1].map1)
                cv_file.write('stereoMapR_y', self.cam_parameters[1].map2)

                cv_file.release()

                
    ##################### Stereopixel to real #############################################################
    def stereopixel_to_real(self, left_xy, right_xy):
        left_xy = np.expand_dims(left_xy, axis=1)
        right_xy = np.expand_dims(right_xy, axis=1)

        undist_left = cv2.fisheye.undistortPoints(left_xy, self.cam_parameters[0].camera_matrix,
                                                self.cam_parameters[0].dist_coeffs,
                                                R=self.cam_parameters[0].r,
                                                P=self.cam_parameters[0].p)

        undist_right = cv2.fisheye.undistortPoints(right_xy, self.cam_parameters[1].camera_matrix,
                                                 self.cam_parameters[1].dist_coeffs,
                                                 R=self.cam_parameters[1].r,
                                                 P=self.cam_parameters[1].p)

        undist_right = undist_right[0]
        undist_left = undist_left[0]

        # Find disparity between corners detected:
        disparity_values = undist_left - undist_right

        # Bundle the values in an array [x, y, d, 1]:
        disparity_points = np.stack((undist_left[:, 0],
                                     undist_left[:, 1],
                                     disparity_values[:, 0],
                                     np.ones([undist_left.shape[0]])), axis=1).T

        # Reproject the bundled points to xyz in the stereo camera coordinate system (left cam):
        reprojection_points = (self.stereo_parameters.q @ disparity_points).T
        real_points = reprojection_points[:, 0:3] / (np.vstack((reprojection_points[:, 3],
                                                                reprojection_points[:, 3],
                                                                reprojection_points[:, 3]))).T
        return real_points, disparity_values

    def undistort_left_2D_points(self, left_xy):
        left_xy = np.expand_dims(left_xy, axis=1)

        undist_left = cv2.fisheye.undistortPoints(left_xy, self.cam_parameters[0].camera_matrix,
                                                  self.cam_parameters[0].dist_coeffs,
                                                  R=self.cam_parameters[0].r,
                                                  P=self.cam_parameters[0].p)
        return undist_left

    def undistort_right_2D_points(self, right_xy):
        right_xy = np.expand_dims(right_xy, axis=1)

        undist_right = cv2.fisheye.undistortPoints(right_xy, self.cam_parameters[0].camera_matrix,
                                                  self.cam_parameters[0].dist_coeffs,
                                                  R=self.cam_parameters[0].r,
                                                  P=self.cam_parameters[0].p)
        return undist_right

    def save_calibration(self, file_path):
        print("Saving parameters!")

        data = {"cam_parameters": self.cam_parameters,
                "stereo_parameters": self.stereo_parameters,
                "image_size": self.image_size}

        with open(file_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
        '''
        ############## SAVE CALIBRATION ###############################################
        print("Saving parameters!")
        data = {"cam_parameters": self.cam_parameters,
                "stereo_parameters": self.stereo_parameters,
                "image_size": self.image_size}

        cv_file = cv2.FileStorage(file_path, cv2.FILE_STORAGE_WRITE)

        # cv_file.write('stereoMapL_x', self.cam_parameters[0].map1)
        # cv_file.write('stereoMapL_y', self.cam_parameters[0].map2)
        # cv_file.write('stereoMapR_x', self.cam_parameters[1].map1)
        # cv_file.write('stereoMapR_y', self.cam_parameters[1].map2)

        cv_file.write("cam_parameters", self.cam_parameters)
        cv_file.write('stereo_parameters', self.stereo_parameters)
        cv_file.write('image_size', self.image_size)

        cv_file.release()
        '''

    def load_calibration(self, file_path):
        '''
        cv_file = cv2.FileStorage()
        cv_file.open(file_path, cv2.FileStorage_READ)

        self.cam_parameters = cv_file.getNode('cam_parameters').mat()
        self.stereo_parameters = cv_file.getNode('stereo_parameters').mat()
        self.image_size = cv_file.getNode('image_size').mat()
        '''
        with open(file_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)

        self.cam_parameters = data["cam_parameters"]
        self.stereo_parameters = data["stereo_parameters"]
        self.image_size = data["image_size"]
        print("         Camera parameters:  ", self.cam_parameters)
        print("         Stereo parameters:  ", self.stereo_parameters)
        print("         Image size:         ", self.image_size)

    def rectify_images(self, left_image, right_image):
        rectified_left_image = cv2.remap(left_image,
                                         self.cam_parameters[0].map1,
                                         self.cam_parameters[0].map2,
                                         cv2.INTER_CUBIC)
        rectified_right_image = cv2.remap(right_image,
                                          self.cam_parameters[1].map1,
                                          self.cam_parameters[1].map2,
                                          cv2.INTER_CUBIC)

        return rectified_left_image, rectified_right_image

    def evaluate_calibration(self, dir_path: str, board_params: tuple, result_path: str):
        print("Stereo manager: Evaluating 3D accuracy")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        images_left = [dir_path + o for o in os.listdir(dir_path) if o.startswith('l_') and o.endswith('.png')]
        images_right = [dir_path + o for o in os.listdir(dir_path) if o.startswith('r_') and o.endswith('.png')]

        images_left.sort()
        images_right.sort()

        objp = np.zeros((board_params[0] * board_params[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_params[0], 0:board_params[1]].T.reshape(-1, 2) * board_params[2]

        point_diff_list = np.zeros([0, 3])

        image_size = cv2.imread(images_left[0], cv2.IMREAD_GRAYSCALE).shape[::-1]

        window_downscale = 3
        window_size = (int(image_size[0] * 2 / window_downscale), int(image_size[1] / window_downscale))
        cv2.namedWindow('Match', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Match', window_size)
     
        for i, fname in enumerate(images_right):
            print(f"     Processing image pair {i + 1} of {len(images_right)} \n", end="")

            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)


            #rect_gray_r, rect_gray_l = self.rectify_images(gray_l, gray_r)
            rect_gray_r, rect_gray_l = calibration.undistortRectify(gray_r, gray_l) # Her kommer det feilmeld


            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(rect_gray_l, (board_params[0], board_params[1]), None)
            ret_r, corners_r = cv2.findChessboardCorners(rect_gray_r, (board_params[0], board_params[1]), None)

            if ret_l and ret_r:
                corners_l = cv2.cornerSubPix(rect_gray_l, corners_l, (11, 11), (-1, -1), criteria) # (11, 11) - winSize: Half of the side length og the search widnow
                corners_r = cv2.cornerSubPix(rect_gray_r, corners_r, (11, 11), (-1, -1), criteria)

                rect_gray_l = cv2.drawChessboardCorners(cv2.cvtColor(rect_gray_l, cv2.COLOR_GRAY2BGR),
                                                        (board_params[0], board_params[1]),
                                                        corners_l,
                                                        ret_l)
                rect_gray_r = cv2.drawChessboardCorners(cv2.cvtColor(rect_gray_r, cv2.COLOR_GRAY2BGR),
                                                        (board_params[0], board_params[1]),
                                                        corners_r,
                                                        ret_r)

                # Find disparity between corners detected:
                disparity_values = corners_l - corners_r

                # Bundle the values in an array [x, y, d, 1]:
                disparity_points = np.stack((corners_l[:, 0, 0],
                                             corners_l[:, 0, 1],
                                             disparity_values[:, 0, 0],
                                             np.ones([corners_l.shape[0]])), axis=1).T

                # Reproject the bundled points to xyz in the stereo camera coordinate system (left cam):
                reprojection_points = (self.stereo_parameters.q @ disparity_points).T
                real_points = reprojection_points[:, 0:3] / (np.vstack((reprojection_points[:, 3],
                                                                        reprojection_points[:, 3],
                                                                        reprojection_points[:, 3]))).T

                # Estimate a transformation matrix between the corners in the board coordinate system and the detected
                # corners in the stereo camera coordinate system:
                t = umeyama(objp.T, real_points.T)

                # Transform the board coordinates to the stereo camera coordinate system:
                estimate_real_points = t @ np.vstack((objp.T, np.ones([objp.shape[0]])))

                # Fins the abs diff between each element in point list of detected corners and actual corners:
                point_diff = real_points - estimate_real_points[0:3, :].T

                point_diff_list = np.vstack((point_diff_list, point_diff))
                print(" - board found")
                cv2.imshow("Match", np.concatenate((rect_gray_l, rect_gray_r), axis=1))
                cv2.waitKey(100)
                
                # Saving images for later use
                cv2.imwrite(result_path + fname[49:], rect_gray_r)
                cv2.imwrite(result_path + "l" + fname[50:], rect_gray_l)
                
                

            else:
                print(" - board not found")
                cv2.imshow("Match", np.concatenate((rect_gray_l, rect_gray_r), axis=1))
                cv2.waitKey(100)

        cv2.destroyAllWindows()

        # Calculate the average abs error for each axis in the stereo camera coordinate system:
        mean_error = np.mean(np.abs(point_diff_list), axis=0)
        error_string = f"Mean abs error: x: {mean_error[0]:.2f} mm, y: {mean_error[1]:.2f} mm, z: {mean_error[2]:.2f} mm"

        print(error_string)

        plt.figure(figsize=(19, 6), dpi=100)
        plt.suptitle(error_string)
        axis = ['X', 'Y', 'Z']
        for i, ax in enumerate(axis):
            plt.subplot(1, 3, i + 1)
            plt.title(f"{ax}-axis Spread: "
                      f"({np.amin(point_diff_list[:, i]):.1f}, "
                      f"{np.amax(point_diff_list[:, i]):.1f}) mm")
            plt.xlabel('Error [mm]')
            plt.ylabel('Frequency')
            plt.text(10, 100, f"")
            plt.hist(x=point_diff_list[:, i], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        #plt.savefig(os.path.join(dir_path, "error_summary1.png"))
        plt.show()

def main():
    result_path = "/home/lotte/TTK4900/chess_frames_padding/chess_calib_results/"   # Path to save result pictures
    stereoMap_path = "/home/lotte/TTK4900/stereoMaps"                               # Path to where you store the map
    img_path = "/home/lotte/TTK4900/chess_frames_padding/chess_calib_img"                          # Path to chessboard pictures

    board_data = (7, 9, 20) # Board real life measurements (Number of squares vertical, Number of squares horizontal, Square length (mm))

    stereo_manager = StereoManager()
    stereo_manager.calibrate_from_dir(img_path, board_data)
    stereo_manager.save_calibration(os.path.join(stereoMap_path, "stereoMap_padding_4.pickle"))
    stereo_manager.evaluate_calibration(img_path, board_data, result_path)
    stereo_manager.load_calibration(os.path.join(stereoMap_path, "stereoMap_padding_4.pickle"))  # 2_stereo_calibration.pickle


if __name__ == '__main__':
    main()

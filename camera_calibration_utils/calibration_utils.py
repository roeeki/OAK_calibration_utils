import os
import numpy as np
import scipy as sp
import cv2

'''
This file holds useful functions for different types of camera calibration
'''

def detect_corners(image_files, calib_board, criteria,
                   sub_pix_criteria=None, sub_pix_win_size=None, sub_pix_zero_zone=None,
                   draw=False):

    object_points = calib_board.get_object_points()

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    non_detected_image_files = []
    img_width = None
    img_height = None
    for file_name in image_files:
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_width = gray.shape[1]
        img_height = gray.shape[0]
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray,
                                                 (calib_board.num_rows, calib_board.num_cols),
                                                 None)

        if (ret == True) and (sub_pix_criteria is not None):
            objpoints.append(object_points)
            corners2 = cv2.cornerSubPix(gray, corners, sub_pix_win_size, sub_pix_zero_zone, sub_pix_criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            if draw:
                cv2.drawChessboardCorners(img, (calib_board.num_rows, calib_board.num_cols), corners2, ret)
                cv2.imshow('corners detection', img)
                cv2.waitKey(500)
        else:
            non_detected_image_files.append(file_name)
    num_images = len(imgpoints)
    print('----- detected corners {} images'.format(num_images))

    cv2.destroyWindow('corners detection')

    # draw non-detection images

    for f in non_detected_image_files:
        img = cv2.imread(f)
        cv2.imshow('undetected images', img)
        cv2.waitKey(500)

    if len(non_detected_image_files) > 0:
        cv2.destroyWindow('undetected images')

    return objpoints, imgpoints, img_width, img_height


def get_images(images_dir, file_suffix=None):
    """
    get all image file names in a folder
    """
    if not os.path.isdir(images_dir):
        Exception('input dir {} not found!'.format(images_dir))

    if file_suffix is None:
        file_suffix_list = ['.jpg', '.png', '.bmp']
    elif isinstance(file_suffix, list):
        file_suffix_list = file_suffix
    elif isinstance(file_suffix, str):
        file_suffix_list = [file_suffix]
    else:
        raise Exception('invalid file suffix!')

    dir_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    image_files = []
    for f in dir_files:
        file_name, fext = os.path.splitext(f)
        if fext not in file_suffix_list:
            continue
        image_files.append(os.path.join(images_dir, f))
    return image_files


def detect_reprojection_outliers(errors_xy, mahalabobis_inlier_threshold=3):
    mahalanobis_distance = mahalanobis(x=np.array(errors_xy))
    is_inlier = mahalanobis_distance <= mahalabobis_inlier_threshold
    return is_inlier


def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : [mxd] query observation vector or matrix.
    data : [nxd] ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
                 If None, data is x
    cov  : [dxd] covariance matrix of the distribution.
                 If None, will be computed from data.
    """
    if not isinstance(x, np.ndarray):
        raise Exception('invalid data size! must be np.ndarray')
    if len(x.shape) == 1:
        x = np.reshape(x, (x.size, 1))
    d = x.shape[1]
    if data is None:
        data=x
    elif data.shape[1] != d:
        raise Exception('invalid data size! must be [mx{}]'.format(d))
    if cov is None:
        cov = np.cov(data.transpose())
    elif cov.shape != (d, d):
        raise Exception('invalid cov size! must be [{}x{}]'.format(d, d))

    x_minus_mu = x - np.mean(data)
    if d == 1:
        inv_covmat = 1/cov
        left_term = np.multiply(x_minus_mu, inv_covmat)
        mahal = np.multiply(left_term, x_minus_mu)
    else:
        inv_covmat = sp.linalg.inv(cov)
        left_term = np.matmul(x_minus_mu, inv_covmat)
        mahal = np.sum(np.multiply(left_term, x_minus_mu), axis=1)  # this is like: left_term * x_minus_mu.T

    return np.sqrt(mahal)

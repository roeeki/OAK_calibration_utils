import os
import cv2
import numpy as np
import yaml


class PinholeCamera:
    """
    Pinhole camera object
    """
    def __init__(self, camera_intrinsic_params_file=None):

        self.id = None

        # camera intrinsic_params
        self.image_size = None
        self.focal_length = None
        self.principal_point = None
        self.radial_distortion_coefficients = None
        self.tangential_distortion_coefficients = None
        self.skew = None
        self.K = None
        self.distortion_coefficients = None

        if camera_intrinsic_params_file is not None:
            self.load(camera_intrinsic_params_file)

    def load(self, camera_intrinsic_params_file):
        """
        load camera intrinsic params from file
        """

        if not os.path.isfile(camera_intrinsic_params_file):
            Exception('camera calibration file: {} not found!'.format(camera_intrinsic_params_file))

        with open(camera_intrinsic_params_file, 'r') as file:
            data = yaml.safe_load(file)

            if 'id' in data.keys():
                self.id = data['id']
            else:
                Exception('id not found!')

            if 'ImageSize' in data.keys():
                self.image_size = np.int32(data['ImageSize'])
                self.image_size = np.flip(self.image_size, axis=None)
            else:
                Exception('PrincipalPoint not found!')

            if 'FocalLength' in data.keys():
                self.focal_length = data['FocalLength']
            else:
                Exception('FocalLength not found!')

            if 'PrincipalPoint' in data.keys():
                self.principal_point = data['PrincipalPoint']
            else:
                Exception('PrincipalPoint not found!')

            if 'Skew' in data.keys():
                self.skew = data['Skew']
            else:
                self.skew = 0

            if 'RadialDistortion' in data.keys():
                self.radial_distortion_coefficients = data['RadialDistortion']
            else:
                self.radial_distortion_coefficients = [0, 0]

            if 'TangentialDistortion' in data.keys():
                self.tangential_distortion_coefficients = data['TangentialDistortion']
            else:
                self.tangential_distortion_coefficients = [0, 0]

            self.K = np.array(((self.focal_length[0], 0, self.principal_point[0]),
                                      (0, self.focal_length[1], self.principal_point[1]),
                                      (0, 0, 1)))
            self.distortion_coefficients = np.concatenate(
                (np.array(self.radial_distortion_coefficients), np.array(self.tangential_distortion_coefficients)))
        return


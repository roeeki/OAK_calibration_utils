import os
import numpy as np
import yaml


class PinholeCamera:
    """
    Pinhole camera object
    """
    def __init__(self, camera_intrinsic_params_file=None):

        # camera name
        self.id = None

        # camera intrinsic_params
        self.image_size = None  # (width, height)
        self.focal_length = None  # (fx, fy)
        self.principal_point = None  # (cx, cy)
        self.skew = None
        self.K = None  # [3x3] intrinsic matrix
        self.dist_coeffs = None  # opencv format: [k1, k2, p1, p2, k3, k4, k5, k6, k7, k8] the rest of the parameters are not supported!

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
                radial_distortion_coefficients = data['RadialDistortion']
            else:
                radial_distortion_coefficients = [0, 0]

            if 'TangentialDistortion' in data.keys():
                tangential_distortion_coefficients = data['TangentialDistortion']
            else:
                tangential_distortion_coefficients = [0, 0]

            self.K = np.array(((self.focal_length[0], 0, self.principal_point[0]),
                                      (0, self.focal_length[1], self.principal_point[1]),
                                      (0, 0, 1)))

            if len(radial_distortion_coefficients)>6:
                raise Exception('only 6 radial distortion coeffs are supported: k1,k2,k3,k4,k5,k6 !')
            if len(tangential_distortion_coefficients)>2:
                raise Exception('only 2 tangential distortion coeffs are supported: p1,p2 !')
            self.dist_coeffs = np.hstack((
                 np.array(radial_distortion_coefficients[:2]),
                 np.array(tangential_distortion_coefficients),
                 np.array(radial_distortion_coefficients[2:]) ))

        return

    def set(self, id, intrinsic_matrix, dist_coeffs, image_size):
        """
        set camera intrinsics
        """
        self.id = id

        if isinstance(intrinsic_matrix, np.ndarray) and intrinsic_matrix.size == 9:
            intrinsic_matrix = np.reshape(intrinsic_matrix, (3,3))
        elif (isinstance(intrinsic_matrix, list) or isinstance(intrinsic_matrix, tuple)) and len(intrinsic_matrix) == 9:
            intrinsic_matrix = np.reshape(np.array(intrinsic_matrix), (3, 3))
        else:
            raise Exception('invalid intrinsic matrix')
        self.K = intrinsic_matrix
        self.focal_length = (intrinsic_matrix[0,0], intrinsic_matrix[1,1])
        self.principal_point = (intrinsic_matrix[0,2], intrinsic_matrix[1,2])
        self.skew = 0

        if isinstance(dist_coeffs, np.ndarray) and dist_coeffs.size <= 8:
            self.dist_coeffs = dist_coeffs.flatten()
        elif (isinstance(dist_coeffs, list) or isinstance(dist_coeffs, tuple)) and len(dist_coeffs) <= 8:
            self.dist_coeffs = np.array(dist_coeffs).flatten()
        else:
            raise Exception('invalid distortion coefficients')

        if isinstance(image_size, np.ndarray) and image_size.size == 2:
            self.image_size = image_size.flatten()
        elif (isinstance(image_size, list) or isinstance(image_size, tuple)) and len(image_size) == 2:
            self.image_size = np.array(image_size).flatten()
        else:
            raise Exception('invalid distortion coefficients')

        return

    def save(self, camera_intrinsics_file):

        output_dir = os.path.dirname(camera_intrinsics_file)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        radial_dist_coeffs = np.hstack([self.dist_coeffs[:2], self.dist_coeffs[4:]])
        tangential_dist_coeffs = self.dist_coeffs[2:4]
        data = {
            'id': self.id,
            'FocalLength': [float(self.K[0, 0]), float(self.K[1, 1])],
            'PrincipalPoint': [float(self.K[0, 2]), float(self.K[1, 2])],
            'ImageSize': [int(self.image_size[0]), int(self.image_size[1])],
            'RadialDistortion': radial_dist_coeffs.flatten().tolist(),
            'TangentialDistortion': tangential_dist_coeffs.flatten().tolist(),
            'Skew': 0}
        with open(camera_intrinsics_file, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=None, sort_keys=False)

        return
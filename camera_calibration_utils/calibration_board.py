import os
import yaml
import numpy as np

class CalibrationBoard:
    """
    This object handles calibration board params
    file format compatible with Kalibr format
    """
    def __init__(self, calibration_board_file):
        """
        load calibration board params from a yaml file
        """
        if not os.path.isfile(calibration_board_file):
            raise Exception('file {} not found!'.format(calibration_board_file))

        with open(calibration_board_file, 'r') as file:
            data = yaml.safe_load(file)

        self.type = data['target_type']  # calibration board type: 'checkerboard' \ 'aprilgrid'
        self.space_units = 'meters'
        if self.type == 'checkerboard':
            self.num_cols = data['targetCols']  # number of internal chessboard corners
            self.num_rows = data['targetRows']  # number of internal chessboard corners
            self.cols_spacing = data['colSpacingMeters']  # size of one chessboard square [m]
            self.rows_spacing = data['rowSpacingMeters']  # size of one chessboard square [m]
            self.tag_size = None  # valid only for aprilgrid
            self.tag_spacing = None  # valid only for aprilgrid

        elif self.type == 'aprilgrid':
            self.num_cols = data['targetCols']  # number of internal chessboard corners
            self.num_rows = data['targetRows']  # number of internal chessboard corners
            self.cols_spacing = None  # valid only for checkerboard
            self.rows_spacing = None  # valid only for checkerboard
            self.tag_size = data['tagSize']  # size of apriltag, edge to edge [m]
            self.tag_spacing = data['tagSpacing']  # ratio of space between tags to tagSize

        else:
            raise Exception('board type {} is ont suppported!'.format(self.board_type))

        return

    def get_object_points(self):
        """
        make object points - opencv compatible
        """

        if self.type == 'checkerboard':
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            num_points = self.num_rows * self.num_cols
            object_points = np.zeros((num_points, 3), np.float32)
            xi = np.linspace(0, (self.num_cols - 1) * self.cols_spacing, self.num_cols)
            yi = np.linspace(0, (self.num_rows - 1) * self.rows_spacing, self.num_rows)
            object_points[:, :2] = np.array(np.meshgrid(xi, yi)).T.reshape(-1, 2)
        else:
            raise Exception('object points for calibration board type: {}, not supported yet!'.format(self.type))
        self.calib_board_object_points = object_points
        self.calib_board_width = self.num_cols
        self.calib_board_height = self.num_rows
        return object_points

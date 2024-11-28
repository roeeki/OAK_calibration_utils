"""Read / Write / analyse record folders. """

import os
import numpy as np
import cv2
from rosbags.highlevel import AnyReader
import pathlib
import matplotlib
# from rosbags.rosbag1 import Writer
# from rosbags.serde import serialize_ros1
# from rosbags.typesys.types import builtin_interfaces__msg__Time as Time
# from rosbags.typesys.types import geometry_msgs__msg__Quaternion as Quaternion
# from rosbags.typesys.types import geometry_msgs__msg__Vector3 as Vector3
# from rosbags.typesys.types import sensor_msgs__msg__Image as Image
# from rosbags.typesys.types import sensor_msgs__msg__Imu as Imu
# from rosbags.typesys.types import std_msgs__msg__Header as Header
from rosbags.typesys import Stores, get_typestore
typestore = get_typestore(Stores.LATEST)
import argparse
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import camera_calibration_utils as ccu


class OakRecordHandler:
    """
    This object handles LULAV test record

    Basic functionality
    1) reads and analyses camera frame and IMU times:
       - checks IMU measurements time synchronization
       - checks camera frame time synchronization
       - checks stereo camera frame synchronization
    2) synchronize camera frames by pwm signal
    """

    def __init__(self, record_folder, plot_results_folder=None):

        self.record_folder = record_folder
        if not os.path.isdir(self.record_folder):
            raise Exception('record folder: {} not found!'.format(self.record_folder))

        self.plot_results_folder = plot_results_folder

        self.matching_frame_times_left = []
        self.matching_frame_times_right = []

        print(f"Reading record...\n")

        left_camera_folder = os.path.join(self.record_folder, 'camera_left')
        if os.path.isdir(left_camera_folder):
            self.left_timestamps_file = os.path.join(left_camera_folder, 'timestamps.txt')
            self.left_images_folder = os.path.join(left_camera_folder, 'images')
        else:
            self.left_timestamps_file = None
            self.left_images_folder = None

        right_camera_folder = os.path.join(self.record_folder, 'camera_right')
        if os.path.isdir(right_camera_folder):
            self.right_timestamps_file = os.path.join(right_camera_folder, 'timestamps.txt')
            self.right_images_folder = os.path.join(right_camera_folder, 'images')
        else:
            self.right_timestamps_file = None
            self.right_images_folder = None

        self.left_frame_times = None
        self.right_frame_times = None
        self._get_camera_frame_times()

        self.imu_file = os.path.join(self.record_folder, 'imu.txt')
        self.imu_times = None
        self.imu_times, _, _ = self._load_imu_data()

        print(f"analysing bag...\n")
        self.analyse_bag_record()
        self.start_time = -np.inf
        self.end_time = np.inf

        self.writer = None

    def _get_camera_frame_times(self):
        """
        get left\right camera frame times and frame ids (used for syncing camera frames)
        check frame ids are unique and sorted for each camera
        """

        # get image messages
        self.left_frame_times, _ = self._load_camera_timestamps(self.left_timestamps_file)
        self.right_frame_times, _ = self._load_camera_timestamps(self.right_timestamps_file)

        # check if sorted and unique
        if self.left_frame_times.size > 0:
            is_sorted = np.all(np.sort(self.left_frame_times) == self.left_frame_times)
            if not is_sorted:
                raise Exception('left frame ids not sorted!')
            is_unique = self.left_frame_times.size == np.unique(self.left_frame_times).size
            if not is_unique:
                raise Exception('left frames not unique!')

        if self.right_frame_times.size > 0:
            is_sorted = np.all(np.sort(self.right_frame_times) == self.right_frame_times)
            if not is_sorted:
                raise Exception('right frame ids not sorted!')
            is_unique = self.right_frame_times.size == np.unique(self.right_frame_times).size
            if not is_unique:
                raise Exception('right frames not unique!')

        return

    def _load_imu_data(self):
        """
        get imu times
        """

        imu_times = []
        acc_data = []
        gyro_data = []
        with open(self.imu_file,'r') as f:
            lines = f.readlines()
            for l in lines:
                if len(l) ==0 or l[0] == '#':
                    pass
                else:
                    sp = l.strip().replace(' ','').replace('\t','').split(',')
                    if len(sp) == 7:
                        imu_times.append(float(sp[0]))
                        acc_data.append((float(sp[1]), float(sp[2]), float(sp[3])))
                        gyro_data.append((float(sp[4]), float(sp[5]), float(sp[6])))
        return np.array(imu_times), np.array(acc_data), np.array(gyro_data)

    @staticmethod
    def _load_camera_timestamps(timestamp_file):
        """
        get left\right camera frame times and frame ids (used for syncing camera frames)
        check frame ids are unique and sorted for each camera
        """

        frame_times = []
        image_files = []
        if timestamp_file is not None:
            with open(timestamp_file,'r') as f:
                lines = f.readlines()
                for l in lines:
                    if len(l) ==0 or l[0] == '#':
                        pass
                    else:
                        sp = l.strip().replace(' ','').replace('\t','').split(',')
                        if len(sp) == 2:
                            frame_times.append(float(sp[0]))
                            image_files.append(sp[1])

        return np.array(frame_times), image_files


    def _sync_camera_frame_times(self):
        """
        sync left\right camera frames by time

        camera sync strategy:
        each frame two frame times:
        drop left/right camera frames without pairs
        """

        mean_step_time = np.mean(self.left_frame_times[1:] - self.left_frame_times[:-1])
        self.matching_frame_times_left = []
        self.matching_frame_times_right = []
        for i, t_left in enumerate(self.left_frame_times):
            d = np.abs(self.right_frame_times - t_left)
            idx = np.argwhere(d <= mean_step_time * 0.25)
            if (idx.size == 1) and (self.right_frame_times[idx.flatten()[0]] not in self.matching_frame_times_right):
                    self.matching_frame_times_left.append(t_left)
                    self.matching_frame_times_right.append(self.right_frame_times[idx.flatten()[0]])

        return

    def analyse_bag_record(self):
        """
        read bag record and analyse timing:
        camera left frames
        camera right frames
        camera left to right frame matches
        IMU measurements
        """

        # analyse image and IMU record statistics
        print('analysing camera \\ imu times ...')

        # check left camera frames
        print('camera frames:')
        if self.left_frame_times.size > 0:
            print('left camera frames:')
            dt_left = self.left_frame_times[1:] - self.left_frame_times[:-1]
            dt_left_mean = np.mean(dt_left)
            dt_left_std = np.std(dt_left)
            print('     {} frames: time range: {:.4f}-{:.4f}, time steps: {:.2f}+-{:.3f}  (mean, std)'.format(self.left_frame_times.size,
                                                                                                              self.left_frame_times[0], self.left_frame_times[-1],
                                                                                                              dt_left_mean, dt_left_std))

        # check right camera frames
        if self.right_frame_times.size > 0:
            print('right camera frames:')
            dt_right = self.right_frame_times[1:] - self.right_frame_times[:-1]
            dt_right_mean = np.mean(dt_right)
            dt_right_std = np.std(dt_right)
            print('     {} frames: time range {:.4f}-{:.4f}, time steps: {:.2f}+-{:.3f}  (mean, std)'.format(self.right_frame_times.size, self.right_frame_times[0], self.right_frame_times[-1], dt_right_mean, dt_right_std))

        # check left-right frame matches
        self._sync_camera_frame_times()
        num_matches = len(self.matching_frame_times_left)
        print('{} valid stereo pairs'.format(num_matches))

        if self.imu_times is not None and self.imu_times.size > 0:
            dt_imu = self.imu_times[1:] - self.imu_times[0:-1]
            dt_imu_mean = np.mean(dt_imu)
            imu_misses = np.sum(
                np.bitwise_or(np.abs(dt_imu) > (dt_imu_mean * 1.5), np.abs(dt_imu) < (dt_imu_mean * 0.5)))
            print('imu: dt = {} +- {} (mean, std)'.format(dt_imu_mean, np.std(dt_imu)))
            print('     time range: {} to {}'.format(dt_imu[0], dt_imu[-1]))
            print('     {} time misses out of {}'.format(imu_misses, dt_imu.shape[0]))
        else:
            print('imu: no measurements found!')

        print('\n')


        if self.plot_results_folder is not None:
            if not os.path.isdir(self.plot_results_folder):
                os.makedirs(self.plot_results_folder)

        dt_left = self.left_frame_times[1:] - self.left_frame_times[0:-1]
        dt_right = self.right_frame_times[1:] - self.right_frame_times[0:-1]
        dt_left_mean = np.mean(dt_left)
        dt_right_mean = np.mean(dt_right)
        fig, ax = plt.subplots(2)
        fig.suptitle('time difference between camera frames', fontsize=15)
        ax[0].scatter(range(0, dt_left.size), dt_left, color=(0, 0, 1), s=2, alpha=1)
        ax[0].set_ylim(bottom=0, top=max(dt_left_mean*2, max(dt_left)))
        ax[0].set_xlabel(r'frame id', fontsize=12)
        ax[0].set_ylabel(r'time difference [sec]', fontsize=12)
        ax[0].set_title('camera left', fontsize=14)
        ax[0].grid(True)

        ax[1].scatter(range(0, dt_right.size), dt_right, color=(0, 0, 1), s=2, alpha=1)
        ax[1].set_ylim(bottom=0, top=max(dt_right_mean*2, max(dt_right)))
        ax[1].set_xlabel(r'frame id', fontsize=12)
        ax[1].set_ylabel(r'time difference [sec]', fontsize=12)
        ax[1].set_title('camera right', fontsize=14)
        ax[1].grid(True)
        # fig.tight_layout()
        plt.draw()
        plt.pause(0.1)

        if self.plot_results_folder is not None:
            camera_time_analysis_file = os.path.join(self.plot_results_folder, 'camera_frames_time_analysis.png')
            fig.savefig(camera_time_analysis_file, bbox_inches='tight')

        dt_imu = self.imu_times[1:] - self.imu_times[0:-1]
        fig2, ax2 = plt.subplots()
        ax2.scatter(range(0, dt_imu.size), dt_imu, color=(0, 0, 1), s=2, alpha=1)
        ax2.set_ylim(bottom=0, top=max(dt_imu_mean*2, max(dt_imu)))
        ax2.set_xlabel(r'measurement id', fontsize=12)
        ax2.set_ylabel(r'time difference [sec]', fontsize=12)
        ax2.set_title('IMU measurement time differences', fontsize=15)
        ax2.grid(True)
        # fig.tight_layout()
        if self.plot_results_folder is not None:
            imu_time_analysis_file = os.path.join(self.plot_results_folder, 'imu_time_analysis.png')
            fig2.savefig(imu_time_analysis_file, bbox_inches='tight')

        # plt.draw()
        # plt.pause(0.1)
        plt.show(block=True)
        print('\n')
        return

    def save_record(self, output_folder, start_time=None, end_time=None, stereo_synchronized_only=False):
        """
        save camera frames and IMU measurements to scenario folder

        time related parameters:
        - save_only_stereo_frames: save only images of stereo matched frames
        - rename_frame_ids_by_stereo_sync: rename right camera frame ids to match the stereo pair left frame ids
                                           (valid only if save_only_stereo_frames is True )
        frame related parameters:
        - use_synced_times: use pwm synchronized frame times
        - start_time, end_time: save only part of the bag record
                                units are in [sec] for start of the record
        """

        print('saving data:')
        print('    output folder: {}'.format(output_folder))

        if start_time is None:
            start_time = -np.inf
        if end_time is None:
            end_time = np.inf
        self.start_time = start_time
        self.end_time = end_time
        print('    taking records from {}[sec] up to {}[sec]'.format(start_time, end_time))

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        with AnyReader([pathlib.Path(self.bag_file)]) as self.bag:

            # save images
            print('   - saving images')
            self._camera_images_to_folder(output_folder, stereo_synchronized_only=stereo_synchronized_only)

            # save IMU data
            print('   - saving imu data')
            imu_folder = os.path.join(output_folder, 'imu0')
            if folder_format.lower() == 'lulav':
                imu_file = os.path.join(imu_folder, 'imu_data.txt')
            elif folder_format.lower() == 'euroc':
                imu_file = os.path.join(imu_folder, 'data.csv')
            else:
                raise Exception('invalid folder format')
            self._save_imu_to_file(imu_file)

            trajectory_file = os.path.join(output_folder, 'trajectory.txt')
            self._trajectory_to_file(trajectory_file, interpulate_to_camera_frames=False)

        print('Done')
        return

    def _camera_images_to_folder(self, output_folder, stereo_synchronized_only=False):
        """
        save camera frames to a folder
        camera_frame_sync: True - use synced times
                           False - use recorded times
        save_only_stereo_frames: True - save only common stereo frames
                                 False - save all frames for each camera
        """

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        timestamp_file_basename = 'timestamps.txt'
        camera_left_subfolder = 'camera_left'
        camera_right_subfolder = 'camera_right'
        image_subfolder_name = 'images'

        camera_folder_left = os.path.join(output_folder, camera_left_subfolder)
        if not os.path.isdir(camera_folder_left):
            os.makedirs(camera_folder_left)
        images_folder_left = os.path.join(camera_folder_left, image_subfolder_name)
        if not os.path.isdir(images_folder_left):
            os.makedirs(images_folder_left)

        camera_folder_right = os.path.join(output_folder, camera_right_subfolder)
        if not os.path.isdir(camera_folder_right):
            os.makedirs(camera_folder_right)
        images_folder_right = os.path.join(camera_folder_right, image_subfolder_name)
        if not os.path.isdir(images_folder_right):
            os.makedirs(images_folder_right)

        # save images
        max_frame_id = max(self.left_frame_times.size, self.right_frame_times.size)
        num_frame_id_digits = int(np.ceil(np.log10(max_frame_id)))

        matching_frame_times_left = np.array(self.matching_frame_times_left)
        matching_frame_times_right = np.array(self.matching_frame_times_right)

        frame_id_left = 0
        timestamps_data = []
        if self.left_camera_image_topic is not None:
            connections = [x for x in self.bag.connections if x.topic == self.left_camera_image_topic]
            for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                msg = self.bag.deserialize(rawdata, connection.msgtype)
                frame_time = float(self.__to_time(msg.header.stamp))

                msg_timestamp = timestamp * 1e-9
                if self.start_time <= msg_timestamp <= self.end_time:
                    d = np.min(np.abs(frame_time - matching_frame_times_left))
                    if (not stereo_synchronized_only) or (d < 1e-5):
                        image_file_name = '{:0{x}d}.png'.format(frame_id_left, x=num_frame_id_digits)
                        frame_id_left += 1
                        timestamps_data.append({'time': frame_time, 'image_file': image_file_name})

                        # write images
                        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                        if im.shape[2] == 1:
                            cv_img = im.squeeze()
                        elif im.shape[2] == 3:
                            # cv_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                            raise Exception('got color image! expecting grayscale!')
                        else:
                            raise Exception('invalid image!')
                        image_file_path = os.path.join(images_folder_left, image_file_name)
                        cv2.imwrite(image_file_path, cv_img)

            # save timestamps file
            camera_left_timestamp_file = os.path.join(camera_folder_left, timestamp_file_basename)
            self._save_camera_timestamps_to_file(camera_left_timestamp_file, timestamps_data)

        frame_id_right = 0
        timestamps_data = []
        if self.right_camera_image_topic is not None:
            connections = [x for x in self.bag.connections if x.topic == self.right_camera_image_topic]
            for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                msg = self.bag.deserialize(rawdata, connection.msgtype)
                frame_time = float(self.__to_time(msg.header.stamp))

                msg_timestamp = timestamp * 1e-9
                if self.start_time <= msg_timestamp <= self.end_time:
                    d = np.min(np.abs(frame_time - matching_frame_times_right))

                    if (not stereo_synchronized_only) or (d < 1e-5):
                        image_file_name = '{:0{x}d}.png'.format(frame_id_right, x=num_frame_id_digits)
                        frame_id_right += 1
                        timestamps_data.append({'time': frame_time, 'image_file': image_file_name})

                        # write images
                        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                        if im.shape[2] == 1:
                            cv_img = im.squeeze()
                        elif im.shape[2] == 3:
                            # cv_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                            raise Exception('got color image! expecting grayscale!')
                        else:
                            raise Exception('invalid image!')
                        image_file_path = os.path.join(images_folder_right, image_file_name)
                        cv2.imwrite(image_file_path, cv_img)

                # save timestamps file
                camera_right_timestamp_file = os.path.join(camera_folder_right, timestamp_file_basename)
                self._save_camera_timestamps_to_file(camera_right_timestamp_file, timestamps_data)

        return

    def _save_camera_timestamps_to_file(self, timestamps_file, timestamps_data):
        """
        save camera frames to a folder
        camera_frame_sync: True - use synced times
                           False - use recorded times
        save_only_stereo_frames: True - save only common stereo frames
                                 False - save all frames for each camera
        """

        output_folder = os.path.dirname(timestamps_file)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        with open(timestamps_file, 'w') as f:
            f.write('#timestamp [ns]    filename\n')
            for ts in timestamps_data:
                f.write('{} {}\n'.format(np.uint64(ts['time'] * 1e9), ts['image_file']))

    def _save_imu_to_file(self, output_file):
        """
        save imu data to a text file
        """

        output_folder = os.path.dirname(output_file)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        with open(output_file, 'w') as f:
            if self.imu_topic is not None:
                f.write('# timestamp [ns]	w_RS_S_x [rad s^-1]	w_RS_S_y [rad s^-1]	w_RS_S_z [rad s^-1]	a_RS_S_x [m s^-2]	a_RS_S_y [m s^-2]	a_RS_S_z [m s^-2]\n')
                connections = [x for x in self.bag.connections if x.topic == self.imu_topic]
                for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                    msg = self.bag.deserialize(rawdata, connection.msgtype)
                    frame_time = float(self.__to_time(msg.header.stamp))
                    msg_timestamp = timestamp * 1e-9
                    if self.start_time <= msg_timestamp <= self.end_time:
                        angular_velocity = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                        linear_acceleration = [msg.linear_acceleration.x, msg.linear_acceleration.y,
                                               msg.linear_acceleration.z]
                        f.write('{} {} {} {} {} {} {}\n'.format(frame_time * 1e9,
                                                                angular_velocity[0], angular_velocity[1],
                                                                angular_velocity[2],
                                                                linear_acceleration[0], linear_acceleration[1],
                                                                linear_acceleration[2]))

        return

    def __del__(self):
        pass


if __name__ == '__main__':

    record_folder = '/home/roee/Projects/OAK_calibration/examples/camera_imu_calibration/OAK_record'
    valid_record_times = {'start': 8, 'end': 32}  # note that this is rosbag record time, and not camera frame times
    save_kalibr_bag = True

    results_folder = './results/camera_imu_calibration'
    plot_res_folder = os.path.join(results_folder,'analysis_results')
    bag_analyzer = OakRecordHandler(record_folder, plot_results_folder=plot_res_folder)

    if save_kalibr_bag:
        left_camera_image_topic = '/cam0/image_raw'
        right_camera_image_topic = '/cam1/image_raw'
        imu_topic = '/imu0'

        # save full bag file: both cameras and IMU with correct timestamps for visual-inertial calibration with Kalibr
        print('\n')
        print('writing full record to Kalibr format:')
        kalibr_bag_file = os.path.join(results_folder,'kalibr_format.bag')
        ccu.save_record_to_bag(kalibr_bag_file, record_folder, camera_left_topic=left_camera_image_topic,
                               camera_right_topic=right_camera_image_topic, imu_topic=imu_topic,
                               start_time=valid_record_times['start'], end_time=valid_record_times['end'])

        # save partial bag file: only left camera with arbitrary timestamps for intrinsic / stereo calibration with Kalibr
        print('\n')
        print('writing only left camera to Kalibr format:')
        kalibr_bag_file2 = os.path.join(results_folder,'kalibr_format2.bag')
        images_folder = os.path.join(record_folder, 'camera_left','images')
        ccu.save_bag_camera_frames(kalibr_bag_file2, images_folder, left_camera_image_topic, 'left', timestamps_files=None,
                                    start_time=None, end_time=None, image_files_suffix='png')

        # save partial bag file: only left+right cameras with arbitrary timestamps for intrinsic / stereo calibration
        print('\n')
        print('writing only left + right camera to Kalibr format:')
        kalibr_bag_file2 = os.path.join(results_folder,'kalibr_format3.bag')
        images_folder = [os.path.join(record_folder, 'camera_left','images'), os.path.join(record_folder, 'camera_right','images')]
        ccu.save_bag_camera_frames(kalibr_bag_file2, images_folder, [left_camera_image_topic, right_camera_image_topic],
                                   ['left','right'], timestamps_files=[None, None],
                                    start_time=None, end_time=None, image_files_suffix='png')

    if plot_res_folder is not None:
        plt.show(block=True)
    print('Done!')

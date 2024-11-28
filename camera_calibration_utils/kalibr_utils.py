"""reformat record to Kalibr bag"""

import os
from os import makedirs

import numpy as np
import cv2
import glob
# from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer
from rosbags.serde import serialize_ros1
from rosbags.typesys.types import builtin_interfaces__msg__Time as Time
from rosbags.typesys.types import geometry_msgs__msg__Quaternion as Quaternion
from rosbags.typesys.types import geometry_msgs__msg__Vector3 as Vector3
from rosbags.typesys.types import sensor_msgs__msg__Image as Image
from rosbags.typesys.types import sensor_msgs__msg__Imu as Imu
from rosbags.typesys.types import std_msgs__msg__Header as Header
from rosbags.typesys import Stores, get_typestore
typestore = get_typestore(Stores.LATEST)


def save_record_to_bag(output_bag_file, record_folder, camera_left_topic=None, camera_right_topic=None, imu_topic=None,
                start_time=None, end_time=None, use_arbitrary_timestamps=False):
    """
    save record to bag file (ros1)
    This is mostly useful for calibrating with Kalibr
    inputs:
    - output_bag_file: bag file path
    - camera_left_topic: left camera topic name. None - don't write this topic
    - camera_right_topic: right camera topic name. None - don't write this topic
    - imu_topic: imu topic name. None - don't write this topic
    """

    if not os.path.isdir(record_folder):
        raise Exception('record folder: {} not found!'.format(record_folder))

    left_camera_folder = os.path.join(record_folder, 'camera_left')
    if os.path.isdir(left_camera_folder):
        left_timestamps_file = os.path.join(left_camera_folder, 'timestamps.txt')
        left_images_folder = os.path.join(left_camera_folder, 'images')
    else:
        left_timestamps_file = None
        left_images_folder = None

    right_camera_folder = os.path.join(record_folder, 'camera_right')
    if os.path.isdir(right_camera_folder):
        right_timestamps_file = os.path.join(right_camera_folder, 'timestamps.txt')
        right_images_folder = os.path.join(right_camera_folder, 'images')
    else:
        right_timestamps_file = None
        right_images_folder = None

    imu_file = os.path.join(record_folder, 'imu.txt')

    output_folder = os.path.dirname(output_bag_file)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    print("writing to bag file: {}".format(output_bag_file))

    # save camera frames to ros bag
    with Writer(output_bag_file) as writer:

        if camera_left_topic is not None:
            print('writing camera left ...')
            save_bag_camera_frames(writer, left_images_folder, camera_left_topic, 'left',
                                   timestamps_files=left_timestamps_file,
                                   start_time=start_time, end_time=end_time)

        if camera_right_topic is not None:
            print('writing camera right ...')
            save_bag_camera_frames(writer, right_images_folder, camera_right_topic, 'right',
                                   timestamps_files=right_timestamps_file,
                                   start_time=start_time, end_time=end_time)

        if imu_topic is not None:
            print('writing imu ...')
            _save_bag_imu(writer, imu_file, imu_topic, start_time=start_time, end_time=end_time)

        print('\n')
        print('-------------------------------------')
        print('to use kalibr camera-imu calibration:')
        print('source ~/kalibr_workspace/devel/setup.bash')
        print('cd ~/kalibr_workspace')
        print('rosrun kalibr kalibr_calibrate_imu_camera --target <target file> --imu <imu_params_file> --imu-models <imu_model> --cam <camera_params_file> --bag <bag file>')
        print('\n')

        print('-------------------------------------')
        print('to use kalibr camera intrinsics / stereo calibration:')
        print('source ~/kalibr_workspace/devel/setup.bash')
        print('cd ~/kalibr_workspace')
        print('rosrun kalibr kalibr_calibrate_cameras --target <target file> --topics <camera topics> --bag <bag file>')
    return

def save_bag_camera_frames(rosbag_writer, images_folders, camera_topics, camera_ids, timestamps_files=None,
                           start_time=None, end_time=None, image_files_suffix='png'):
    """
    save camera frames to ros bag

    inputs:
    rosbag_writer - rosbags writer object (if you want to add to existing writer)
                    rosbags file name if you want a new writer
    images_folder - folder with images
    timestamps_file - timestamps file. if None, arbitrary timestamps will be used!
    camera_topic - camera topic name
    camera_id - camera name (string)
    start_time, end_time - save only part of the recording
    image_files_suffix - only relevant when not using timestamps file
    """

    # handle multiple cameras Vs single camera
    if not isinstance(images_folders, list):
        images_folders = [images_folders]
    if not isinstance(camera_topics, list):
        camera_topics = [camera_topics]
    if not isinstance(camera_ids, list):
        camera_ids = [camera_ids]
    if not isinstance(timestamps_files, list):
        timestamps_files = [timestamps_files]

    # check all inputs are same size
    n = len(images_folders)
    if n != len(camera_topics) or n != len(camera_ids) or n != len(timestamps_files):
        raise Exception('invalid input! not same number of cameras!')

    # handle rosbags writer
    # to add to an existing rosbags instance - use an open writer
    # to write only camera frames you can use a filename
    close_writer = False
    if isinstance(rosbag_writer, Writer):
        pass
    else:
        try:
            folder_path = os.path.dirname(rosbag_writer)
            if not os.path.isdir(folder_path):
                makedirs(folder_path)
            rosbag_writer = Writer(rosbag_writer)
            rosbag_writer.open()
            close_writer = True
        except:
            raise Exception('invalid rosbag writer! must be non exsisting file path or rosbags.rosbag1.write.Writer')

    for i in range(n):
        images_folder = images_folders[i]
        camera_topic = camera_topics[i]
        camera_id = camera_ids[i]
        timestamps_file = timestamps_files[i]

        if (timestamps_file is not None) and (not os.path.isfile(timestamps_file)):
            raise Exception('camera timestamps file not found: {}'.format(timestamps_file))
        if os.path.isdir(images_folder) is None:
            raise Exception('image folder not found: {}'.format(images_folder))
        if camera_topic[0] != '/':
            raise Exception('invalid camera topic: {} - missing starting /'.format(camera_topic))
        if start_time is None:
            start_time = -np.inf
        if end_time is None:
            end_time = np.inf

        if timestamps_file is None:
            print('camera {}: no timestamps file! using arbitrary timestamps!'.format(camera_id))

        conn = rosbag_writer.add_connection(camera_topic, Image.__msgtype__, typestore=typestore)
        frame_id = 0

        if timestamps_file is not None:  # use timestamps file to get images and timestamps
            with open(timestamps_file, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    time_stamp = None
                    image_file = None
                    if len(l) == 0 or l[0] == '#':
                        pass
                    else:
                        sp = l.strip().replace(' ','').replace('\t','').split(',')
                        if len(sp) == 2:
                            time_stamp = float(sp[0])
                            image_file = os.path.join(images_folder, sp[1])

                    if (time_stamp is not None) and (start_time <= time_stamp <= end_time):
                        _write_frame(rosbag_writer, conn, image_file, time_stamp, camera_id, draw=True, window_name='img')
                        frame_id = frame_id + 1

        else:  # write all images in folder with arbitrary timestamps
            image_files = sorted(glob.glob(os.path.join(images_folder,'*.{}'.format(image_files_suffix))))
            if len(image_files) == 0:
                raise Exception('no images were found in: {}'.format(os.path.join(images_folder,'*.{}'.format(image_files_suffix))))
            for image_file in image_files:
                time_stamp = float(frame_id*0.05)  # frequency is 20Hz, just for easy display in foxglove
                _write_frame(rosbag_writer, conn, image_file, time_stamp, camera_id, draw=True, window_name='img')
                frame_id = frame_id + 1

    if close_writer:
        rosbag_writer.close()


def _write_frame(rosbag_writer, rosbag_conn, image_file, time_stamp, camera_id, draw=False, window_name='img'):

    img = cv2.imread(image_file)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_width = int(img.shape[1])
    img_height = int(img.shape[0])
    timestamp = Time(sec=int(time_stamp // 1), nanosec=int((time_stamp % 1) * 1e9))

    if draw:
        cv2.imshow(window_name, img)
        cv2.waitKey(50)

    # data = np.fromfile(image_file, dtype=np.uint8)
    data = np.frombuffer(img, dtype=np.uint8)
    msg_ros1 = Image(
        Header(stamp=timestamp, frame_id=camera_id),
        height=img_height,  # 224
        width=img_width,  # 225
        encoding='8UC1',  # 'mono8',  # 8bit
        is_bigendian=False,  # False
        step=img_width,  # 225
        # data=ii,  # ii.size = 50400=hxw, ERROR!!!
        # ValueError: memoryview assignment: lvalue and rvalue have different structures
        data=data,
        # dd.size = 44157=png-file size, # Runs, but bag fails to deserialize image in rviz
    )
    rawdata_ros1 = typestore.serialize_ros1(msg_ros1, Image.__msgtype__)
    rosbag_writer.write(rosbag_conn, int(time_stamp * 10e9), rawdata_ros1)
    return


def _save_bag_imu(rosbag_writer, imu_file, imu_topic, start_time=None, end_time=None):
    """
    save camera frames to ros bag
    """

    # save ros bag
    if rosbag_writer is None:
        raise Exception('bag writer not valid!')
    if os.path.isfile(imu_file) is None:
        raise Exception('imu file not found: {}'.format(imu_file))
    if imu_topic[0] != '/':
        raise Exception('invalid imu topic: {} - missing starting /'.format(imu_topic))
    if start_time is None:
        start_time = -np.inf
    if end_time is None:
        end_time = np.inf

    conn = rosbag_writer.add_connection(imu_topic, Imu.__msgtype__, typestore=typestore)
    frame_id = np.uint64(0)
    with open(imu_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            time_stamp = None
            image_file = None
            if len(l) == 0 or l[0] == '#':
                pass
            else:
                sp = l.strip().replace(' ','').replace('\t','').split(',')
                if len(sp) == 7:
                    time_stamp = float(sp[0])
                    orientation = np.array((1,0,0,0))
                    orientation_covariance = np.zeros((3, 3), dtype=float)
                    angular_velocity = np.array((float(sp[4]), float(sp[5]), float(sp[6])), dtype=float)
                    angular_velocity_covariance = np.zeros((3, 3), dtype=float)
                    linear_acceleration = np.array((float(sp[1]), float(sp[2]), float(sp[3])), dtype=float)
                    linear_acceleration_covariance = np.zeros((3, 3), dtype=float)

            if (time_stamp is not None) and (start_time <= time_stamp <= end_time):
                timestamp = Time(sec=int(time_stamp // 1), nanosec=int((time_stamp % 1)*1e9))
                msg_ros1 = Imu(
                    Header(stamp = timestamp, frame_id = 'imu'),
                    orientation = Quaternion(w=orientation[0], x=orientation[1], y=orientation[2], z=orientation[3]),
                    orientation_covariance = orientation_covariance.flatten(),
                    angular_velocity = Vector3(x=angular_velocity[0], y=angular_velocity[1], z=angular_velocity[2]),
                    angular_velocity_covariance = angular_velocity_covariance.flatten(),
                    linear_acceleration = Vector3(x=linear_acceleration[0], y=linear_acceleration[1], z=linear_acceleration[2]),
                    linear_acceleration_covariance = linear_acceleration_covariance.flatten()
                )
                rawdata_ros1 = typestore.serialize_ros1(msg_ros1, Imu.__msgtype__)
                rosbag_writer.write(conn, int(time_stamp*10e9), rawdata_ros1)
                frame_id = frame_id + 1

    return

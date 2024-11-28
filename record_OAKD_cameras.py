#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import os
import datetime
from pynput import keyboard
import threading
import time

snapshot_on = False
video_record = False
c = threading.Condition()


class FrameRecorder:
    """
    FrameGrabber is saves camera frames to LULAV-NAV standard record folder:

    +-- record_folder
        |-- record_setup.txt
        |-- imu.txt
        +-- camera_left
            |-- timestamps.txt
            +-- images
                |-- 00000.png
                |--   ...
                |-- 00042.png
        +-- camera_left
            |-- timestamps.txt
            +-- images
                |-- 00000.png
                |--   ...
                |-- 00042.png

    timestamps.txt file format:
    # timestamp  image_file
    0.0    00000.png
    0.05   00001.png
    0.10   00003.png
    ...

    """

    def __init__(self, output_folder):

        self.camera_left_output_folder = os.path.join(output_folder, 'camera_left')
        self.camera_right_output_folder = os.path.join(output_folder, 'camera_right')
        self.image_left_output_folder = os.path.join(self.camera_left_output_folder, 'images')
        self.image_right_output_folder = os.path.join(self.camera_right_output_folder, 'images')
        if not os.path.isdir(self.image_left_output_folder):
            os.makedirs(os.path.join(self.image_left_output_folder))
        if not os.path.isdir(self.image_right_output_folder):
            os.makedirs(os.path.join(self.image_right_output_folder))

        self.camera_left_timestamps_file = os.path.join(self.camera_left_output_folder, 'timestamps.txt')
        self.camera_right_timestamps_file = os.path.join(self.camera_right_output_folder, 'timestamps.txt')
        self.f_left = open(self.camera_left_timestamps_file, "w", buffering=1)  # flush write buffer to file every line
        self.f_left.write("# timestamp, image file\n")
        self.f_right = open(self.camera_right_timestamps_file, "w", buffering=1)  # flush write buffer to file every line
        self.f_right.write("# timestamp, image file\n")
        print('writing results to: {}'.format(output_folder))

    def save_frame_left(self, image, timestamp, frame_id):
        img_left_file_name = '{:05d}.png'.format(frame_id)
        img_left_file_path = os.path.join(self.image_left_output_folder, img_left_file_name)
        cv2.imwrite(img_left_file_path, image)
        self.f_left.write('{:f}, {}\n'.format(timestamp, img_left_file_name))

    def save_frame_right(self, image, timestamp, frame_id):
        img_right_file_name = '{:05d}.png'.format(frame_id)
        img_right_file_path = os.path.join(self.image_right_output_folder, img_right_file_name)
        cv2.imwrite(img_right_file_path, image)
        self.f_right.write('{:f}, {}\n'.format(timestamp, img_right_file_name))

    def __del__(self):
        self.f_left.close()
        print('closed f_left')
        self.f_right.close()
        print('closed f_right')

def get_cameras(queue_left, queue_right, img_size, output_folder, plot):
    """
    This function is the get camera frames thread main function

    :param queue_left: left OAK camera queue
    :param queue_right: right OAK camera queue
    :param img_size: OAK frame size
    :param output_folder: output record folder
    :param plot:
    :return:
    """
    global snapshot_on
    global video_record

    # setup frame recorder
    fg = FrameRecorder(output_folder)

    # setup draw
    frame_left = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
    frame_right = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
    frame_left_rs = np.zeros((int(img_size[0] / 2), int(img_size[1] / 2), 3), dtype=np.uint8)
    frame_right_rs = np.zeros((int(img_size[0] / 2), int(img_size[1] / 2)), dtype=np.uint8)
    frame_id = 0
    got_frame_left = False
    got_frame_right = False
    plot_frame_id = False

    while True:
        # get left frame
        inLeft = queue_left.tryGet()
        if inLeft is not None:
            frame_left = inLeft.getCvFrame()
            frame_left_time = inLeft.getRaw().tsDevice
            got_frame_left = True
        else:
            got_frame_left = False

        # get right frame
        inRight = queue_right.tryGet()
        if inRight is not None:
            frame_right = inRight.getCvFrame()
            frame_right_time = inRight.getRaw().tsDevice
            got_frame_right = True
        else:
            got_frame_right = False

        # handle record mode:
        # - video mode is initiated by user, and automatically turned off after one frame
        # - video mode is fully controlled by user
        # save frames if snapshot or video mode.
        if snapshot_on or video_record:
            if got_frame_left and got_frame_right:
                if snapshot_on:
                    print('snapshot {}'.format(frame_id))
                    c.acquire()
                    snapshot_on = False
                    print('get_cameras: snapshot_on release')
                    c.release()
                    plot_frame_id = True

                else:
                    plot_frame_id = False

                fg.save_frame_left(frame_left, frame_left_time, frame_id)
                fg.save_frame_right(frame_right, frame_right_time, frame_id)
                frame_id = frame_id + 1

        if plot:
            frame_left_rs[:, :, 0] = cv2.resize(frame_left, [320, 200])
            frame_left_rs[:, :, 1] = cv2.resize(frame_left, [320, 200])
            frame_left_rs[:, :, 2] = cv2.resize(frame_left, [320, 200])
            if video_record:
                cv2.circle(frame_left_rs, (20, 20), 6, (70, 70, 255), 10)
            if plot_frame_id:
                frame_left_rs = cv2.putText(frame_left_rs, '{}'.format(frame_id), (40, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (70, 70, 255), 2, cv2.LINE_AA)
            cv2.imshow("left image", frame_left_rs)

            frame_right_rs = cv2.resize(frame_right, [320, 200])
            cv2.imshow("right image", frame_right_rs)
            cv2.waitKey(10)

            if plot_frame_id:
                plot_frame_id = False
                time.sleep(0.2)

def log_record_setup(log_file, record_time=None, device_id=None, camera_left_params=None, camera_right_params=None):
    """
    log major attributes of the record in the beginning of the session
    """
    output_folder = os.path.dirname(log_file)
    if not os.path.isdir(output_folder):
        os.makedirs(os.path.join(output_folder))

    with open(log_file, 'w') as log_file:
        if record_time is not None:
            log_file.write('record start time: {}\n\n'.format(record_time))
        if device_id is not None:
            log_file.write('OAK device MxId: {}\n\n'.format(device_id))
        if camera_left_params is not None:
            log_file.write('camera left: {}\n'.format(camera_left_params['name']))
            log_file.write('     board socket: {}\n'.format(camera_left_params['board_socket']))
            log_file.write('     fps: {}\n'.format(camera_left_params['fps']))
            log_file.write('     resolution:  {} = ({} x {})\n'.format(camera_left_params['resolution_name'], camera_left_params['resolution_width'], camera_left_params['resolution_height']))
            log_file.write('     Intrinsics: {}\n'.format(camera_left_params['intrinsics'][0]))
            log_file.write('                 {}\n'.format(camera_left_params['intrinsics'][1]))
            log_file.write('                 {}\n'.format(camera_left_params['intrinsics'][2]))
            log_file.write('\n')

        if camera_right_params is not None:
            log_file.write('camera right: {}\n'.format(camera_right_params['name']))
            log_file.write('     board socket: {}\n'.format(camera_right_params['board_socket']))
            log_file.write('     fps: {}\n'.format(camera_left_params['fps']))
            log_file.write('     resolution:  {} = ({} x {})\n'.format(camera_right_params['resolution_name'], camera_right_params['resolution_width'], camera_right_params['resolution_height']))
            log_file.write('     Intrinsics: {}\n'.format(camera_right_params['intrinsics'][0]))
            log_file.write('                 {}\n'.format(camera_right_params['intrinsics'][1]))
            log_file.write('                 {}\n'.format(camera_right_params['intrinsics'][2]))
            log_file.write('\n')

# callback for key presses, the listener will pass us a key object that
# indicates what key is being pressed
def on_key_press(key):
    """
    get key presses from the user, and handle global snapshot / video flags accordingly
    """
    global snapshot_on
    global video_record

    print("Key pressed: ", key)
    # so this is a bit of a quirk with pynput,
    # if an alpha-numeric key is pressed the key object will have an attribute
    # char which contains a string with the character, but it will only have
    # this attribute with alpha-numeric, so if a special key is pressed
    # this attribute will not be in the object.
    # so, we end up having to check if the attribute exists with the hasattr
    # function in python, and then check the character
    # here is that in action:
    if (hasattr(key, "char") and key.char == "s") or key == keyboard.Key.space:
        c.acquire()
        snapshot_on = True
        print('get_key: snapshot_on={}'.format(snapshot_on))
        c.release()

    if hasattr(key, "char") and key.char == "v":
        c.acquire()
        video_record = not video_record
        print('get_key: video_record={}'.format(video_record))
        c.release()


if __name__ == "__main__":
    results_folder = './records'
    plot = True
    camera_fps = 20

    now = datetime.datetime.now()
    res_folder = os.path.join(results_folder, now.strftime("%Y%m%d_%H%M%S"))

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    xout = pipeline.create(dai.node.XLinkOut)
    xoutLeft = pipeline.create(dai.node.XLinkOut)
    xoutRight = pipeline.create(dai.node.XLinkOut)

    xout.setStreamName("disparity")
    xoutLeft.setStreamName("left")
    xoutRight.setStreamName("right")

    # Properties
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoLeft.setFps(camera_fps)
    monoRight.setFps(camera_fps)

    # Linking
    monoLeft.out.link(xoutLeft.input)
    monoRight.out.link(xoutRight.input)


    # frame_full = np.zeros((full_frame_height, full_frame_width, 3), dtype=np.uint8)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # record setup file
        record_setup_log_file = os.path.join(res_folder, 'record_setup_file.txt')

        calibData = device.readCalibration()
        intrinsics_left = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)
        intrinsics_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C)

        record_start_time =  now.strftime("%Y%m%d_%H%M%S")
        OAK_device_id = device.getMxId()
        camera_left_params = {'name': monoLeft.getName(), 'board_socket': monoLeft.getBoardSocket(),
                              'fps': monoLeft.getFps(),
                              'resolution_name': monoLeft.getResolution().name,
                              'resolution_width': monoLeft.getResolutionWidth(),
                              'resolution_height': monoLeft.getResolutionHeight(),
                              'intrinsics': intrinsics_left}
        camera_right_params = {'name': monoRight.getName(), 'board_socket': monoRight.getBoardSocket(),
                              'fps': monoRight.getFps(),
                              'resolution_name': monoRight.getResolution().name,
                              'resolution_width': monoRight.getResolutionWidth(),
                              'resolution_height': monoRight.getResolutionHeight(),
                              'intrinsics': intrinsics_right}

        log_record_setup(record_setup_log_file, record_time=record_start_time, device_id=OAK_device_id,
                         camera_left_params=camera_left_params, camera_right_params=camera_right_params)

        # create a listener and setup our call backs
        keyboard_listener = keyboard.Listener(
            on_press=on_key_press)

        # create get camera thread
        ql = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        qr = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        gray_frame_size = [monoLeft.getResolutionHeight(), monoLeft.getResolutionWidth()]
        t1 = threading.Thread(target=get_cameras, args=(ql, qr, gray_frame_size, res_folder, True))

        print("starting the camera thread...")
        t1.start()
        print("starting the keyboard listener...")
        keyboard_listener.start()

        t1.join()
        keyboard_listener.join()

    print('Done')

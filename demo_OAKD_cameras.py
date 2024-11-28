#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np


if __name__ == "__main__":

    camera_fps = 20

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    # xout = pipeline.create(dai.node.XLinkOut)
    xoutLeft = pipeline.create(dai.node.XLinkOut)
    xoutRight = pipeline.create(dai.node.XLinkOut)

    # xout.setStreamName("disparity")
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

    # scenario draw setup
    gray_frame_size = [monoLeft.getResolutionHeight(), monoLeft.getResolutionWidth()]
    frame_left = np.zeros((gray_frame_size[0], gray_frame_size[1]), dtype=np.uint8)
    frame_right = np.zeros((gray_frame_size[0], gray_frame_size[1]), dtype=np.uint8)
    frame_left_rs = np.zeros((int(gray_frame_size[0] / 2), int(gray_frame_size[1] / 2), 3), dtype=np.uint8)
    frame_right_rs = np.zeros((int(gray_frame_size[0] / 2), int(gray_frame_size[1] / 2)), dtype=np.uint8)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # Output queue will be used to get the disparity frames from the outputs defined above
        ql = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        qr = device.getOutputQueue(name="left", maxSize=4, blocking=False)

        calibData = device.readCalibration()
        intrinsics_left = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)
        intrinsics_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C)

        print('------------------------------------------')
        print('device MxId: {}'.format(device.getMxId()))
        print('------------------------------------------\n')
        print('camera left setup: {}'.format(monoLeft.getName()))
        print('   Board socket: {}'.format(monoLeft.getBoardSocket()))
        print('   Resolution:  {}=({} X {})'.format(monoLeft.getResolution().name, monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight()))
        print('   Frame rate:   {}'.format(monoLeft.getFps()))
        print('   Intrinsics:   {}'.format(intrinsics_left[0]))
        print('                 {}'.format(intrinsics_left[1]))
        print('                 {}\n'.format(intrinsics_left[2]))

        print('camera right setup: {}'.format(monoRight.getName()))
        print('   Board socket: {}'.format(monoRight.getBoardSocket()))
        print('   Resolution:  {}=({} X {})'.format(monoRight.getResolution().name, monoRight.getResolutionWidth(), monoRight.getResolutionHeight()))
        print('   Frame rate:   {}'.format(monoRight.getFps()))
        print('   Intrinsics:   {}'.format(intrinsics_right[0]))
        print('                 {}'.format(intrinsics_right[1]))
        print('                 {}\n'.format(intrinsics_right[2]))

        while True:

            inLeft = ql.tryGet()
            if inLeft is not None:
                frame_left = inLeft.getCvFrame()
                frame_left_rs = cv2.resize(frame_left, [320, 200])
                cv2.imshow("left image", frame_left_rs)
                cv2.waitKey(10)

            inRight = qr.tryGet()
            if inRight is not None:
                frame_right = inRight.getCvFrame()
                frame_right_rs = cv2.resize(frame_right, [320, 200])
                cv2.imshow("right image", frame_right_rs)
                cv2.waitKey(10)

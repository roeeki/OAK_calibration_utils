import threading
import depthai as dai
import numpy as np
import cv2
import datetime
import os

import record_OAKD_cameras as rc
import record_OAKD_imu as ri


def get_cameras(queue_left, queue_right, img_size, output_folder, plot=True):

    print('Starting camera thread...')
    fg = rc.FrameRecorder(output_folder)

    # scenario draw setup
    frame_left = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
    frame_right = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
    frame_left_rs = np.zeros((int(img_size[0] / 2), int(img_size[1] / 2)), dtype=np.uint8)
    frame_right_rs = np.zeros((int(img_size[0] / 2), int(img_size[1] / 2)), dtype=np.uint8)

    frame_id = -1
    got_frame_left = False
    got_frame_right = False

    while True:
        inLeft = queue_left.tryGet()
        if inLeft is not None:
            frame_left = inLeft.getCvFrame()
            frame_left_time = inLeft.getRaw().tsDevice
            got_frame_left = True
        else:
            got_frame_left = False

        inRight = queue_right.tryGet()
        if inRight is not None:
            frame_right = inRight.getCvFrame()
            frame_right_time = inRight.getRaw().tsDevice
            got_frame_right = True
        else:
            got_frame_right = False

        if got_frame_left or got_frame_right:
            frame_id = frame_id + 1
            snapshot_on = False
        if got_frame_left:
            fg.save_frame_left(frame_left, frame_left_time, frame_id)
            # print('image_left at: {}'.format(frame_left_time))

        if got_frame_right:
            fg.save_frame_right(frame_right, frame_right_time, frame_id)
            # print('image_right at: {}'.format(frame_right_time))

        if plot:
            frame_left_rs = cv2.resize(frame_left, [320, 200])
            cv2.imshow("left image", frame_left_rs)
            frame_right_rs = cv2.resize(frame_right, [320, 200])
            cv2.imshow("right image", frame_right_rs)
            cv2.waitKey(1)


# # custom thread class
# class RecordImuThread(threading.Thread):
#     # override the constructor
#     def __init__(self, imuQueue, output_folder=None):
#         # execute the base constructor
#         threading.Thread.__init__(self)
#         # store the value
#         self.imuQueue = imuQueue
#         self.output_folder = output_folder
#         self.f = None
#
#     # override the run function
#     def run(self):
#          # TODO: make this in to a Thread class, and close IMU file properly
#         print('Starting imu thread...')
#         ig = ri.ImuRecorder(self.output_folder)
#
#         if self.output_folder is not None:
#             imu_file = os.path.join(self.output_folder,'imu.txt')
#             with open(imu_file, "w") as self.f:
#                 self.f.write("#  time, acc(x, y, z), gyro(x, y, z)\n")
#                 # baseTs = None
#                 while True:
#                     imuData = self.imuQueue.get()  # blocking call, will wait until a new data has arrived
#                     imuPackets = imuData.packets
#                     for imuPacket in imuPackets:
#                         acceleroValues = imuPacket.acceleroMeter
#                         gyroValues = imuPacket.gyroscope
#                         acceleroTs = acceleroValues.getTimestampDevice()
#                         gyroTs = gyroValues.getTimestampDevice()
#                         # if baseTs is None:
#                         #     baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs
#
#                         ts = min(acceleroTs.total_seconds(), gyroTs.total_seconds())
#                         ig.save(ts, acceleroValues, gyroValues)


def get_imu(imuQueue, output_folder=None):
    # TODO: make this in to a Thread class, and close IMU file properly
    print('Starting imu thread...')
    ig = ri.ImuRecorder(output_folder)

    if output_folder is not None:
        imu_file = os.path.join(output_folder,'imu.txt')
        f = open(imu_file, "w")
        f.write("#  time, acc(x, y, z), gyro(x, y, z)\n")
        f.close()

        while True:
            imuData = imuQueue.get()  # blocking call, will wait until a new data has arrived
            # imuData = imuQueue.tryGet()
            if imuData is not None:
                imuPackets = imuData.packets
                for imuPacket in imuPackets:
                    acceleroValues = imuPacket.acceleroMeter
                    gyroValues = imuPacket.gyroscope
                    acceleroTs = acceleroValues.getTimestampDevice()
                    gyroTs = gyroValues.getTimestampDevice()
                    # if baseTs is None:
                    #     baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs

                    ts = min(acceleroTs.total_seconds(), gyroTs.total_seconds())
                    ig.save(ts, acceleroValues, gyroValues)

                    # acceleroTs = acceleroTs - baseTs
                    # gyroTs = gyroTs - baseTs
                    # print("Accelerometer timestamp: {:.06f} ms".format(acceleroTs.total_seconds()*1000))
                    # print("Accelerometer [m/s^2]: x: {:.06f} y: {:.06f} z: {:.06f}"
                    #       .format(acceleroValues.x, acceleroValues.y, acceleroValues.z))
                    #
                    # print("Gyroscope timestamp: {:.03f} ms".format(gyroTs.total_seconds()*1000))
                    # print("Gyroscope [rad/s]: x: {:.03f} y: {:.03f} z: {:.03f} "
                    #       .format(gyroValues.x, gyroValues.y, gyroValues.z))


if __name__ =="__main__":

    results_folder = './records'
    now = datetime.datetime.now()
    output_folder = os.path.join(results_folder, now.strftime("%Y%m%d_%H%M%S"))

    if not os.path.isdir(results_folder):
        os.makedirs(os.path.join(results_folder))
    if not os.path.isdir(output_folder):
        os.makedirs(os.path.join(output_folder))

    # Create pipeline
    pipeline = dai.Pipeline()

    # --------------------------- setup camera connection -------------------------------
    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    xoutLeft = pipeline.create(dai.node.XLinkOut)
    xoutRight = pipeline.create(dai.node.XLinkOut)
    xoutLeft.setStreamName("left")
    xoutRight.setStreamName("right")

    # Properties
    monoLeft.setFps(20)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setFps(20)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # Linking
    monoLeft.out.link(xoutLeft.input)
    monoRight.out.link(xoutRight.input)
    gray_frame_size = [monoLeft.getResolutionHeight(), monoLeft.getResolutionWidth()]


    # --------------------------- setup IMU connection -------------------------------
    # Define sources and outputs
    imu = pipeline.create(dai.node.IMU)
    xlinkOut = pipeline.create(dai.node.XLinkOut)
    xlinkOut.setStreamName("imu")
    # enable ACCELEROMETER_RAW at 500 hz rate
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 250)  # 15Hz, 31Hz, 62Hz, 125Hz, 250Hz 500Hz
    # enable GYROSCOPE_RAW at 400 hz rate
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 200)  # 25Hz, 33Hz, 50Hz, 100Hz, 200Hz, 400Hz
    # it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
    # above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
    imu.setBatchReportThreshold(20)
    # maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
    # if lower or equal to batchReportThreshold then the sending is always blocking on device
    # useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
    imu.setMaxBatchReports(20)
    # Link plugins IMU -> XLINK
    imu.out.link(xlinkOut.input)


    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        print('------------------------------------------')
        print('device MxId: {}'.format(device.getMxId()))
        print('------------------------------------------\n')

        # Output queue will be used to get the disparity frames from the outputs defined above
        queue_left = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        queue_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        queue_imu = device.getOutputQueue(name="imu", maxSize=50, blocking=False)

        t1 = threading.Thread(target=get_cameras, args=(queue_left, queue_right, gray_frame_size, output_folder, True))
        t2 = threading.Thread(target=get_imu, args=(queue_imu, output_folder))
        # t2 = RecordImuThread(queue_imu, output_folder)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

    print("Done!")


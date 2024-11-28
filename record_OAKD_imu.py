#!/usr/bin/env python3
import depthai as dai
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib
import time


class ImuRecorder:
    def __init__(self, output_folder):

        self.output_folder = output_folder
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        self.imu_file = os.path.join(output_folder, 'imu.txt')
        self.f_imu = open(self.imu_file, "w", buffering=1)  # flush write buffer to file every line
        self.f_imu.write("# timestamp, acc(X,Y,Z), gyro(X,Y,Z)\n")
        print('writing results to: {}'.format(self.imu_file))

    def save(self, timestamp, acc, gyro):
        self.f_imu.write('{:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}\n'.format(timestamp,
                                                                             acc.x, acc.y, acc.z,
                                                                             gyro.x, gyro.y, gyro.z))
    def __del__(self):
        self.f_imu.close()
        print('closed f_imu')


if __name__ == "__main__":

    results_folder = './records'
    plot = True
    num_measurements_to_plot = 100

    # matplotlib.rcParams['interactive'] == True
    matplotlib.interactive(True)

    now = datetime.datetime.now()
    res_folder = os.path.join(results_folder, now.strftime("%Y%m%d_%H%M%S"))
    ig = ImuRecorder(res_folder)

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    imu = pipeline.create(dai.node.IMU)
    xlinkOut = pipeline.create(dai.node.XLinkOut)

    xlinkOut.setStreamName("imu")

    # enable ACCELEROMETER_RAW at 500 hz rate
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 200)
    # enable GYROSCOPE_RAW at 400 hz rate
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 200)
    # it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
    # above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
    imu.setBatchReportThreshold(1)
    # maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
    # if lower or equal to batchReportThreshold then the sending is always blocking on device
    # useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
    imu.setMaxBatchReports(10)

    # Link plugins IMU -> XLINK
    imu.out.link(xlinkOut.input)

    acc_data = np.zeros((num_measurements_to_plot, 3), dtype=float)
    gyr_data = np.zeros((num_measurements_to_plot, 3), dtype=float)
    timestamps = np.zeros((num_measurements_to_plot,1), dtype=float)

    fig, ax = plt.subplots()
    graph1 = ax.plot(timestamps, acc_data[:, 0], 'r')
    graph2 = ax.plot(timestamps, acc_data[:, 1], 'b')
    graph3 = ax.plot(timestamps, acc_data[:, 2], 'g')
    ax.set_xlabel("time [sec]")
    ax.set_ylabel("acceleration [n/sec^2]")
    ax.set_title('accelerometer data')
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.0001)

    # Pipeline is defined, now we can connect to the device
    with dai.Device(pipeline) as device:

        def timeDeltaToMilliS(delta) -> float:
            return delta.total_seconds() * 1000

        # Output queue for imu bulk packets
        imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
        baseTs = None
        while True:
            imuData = imuQueue.get()  # blocking call, will wait until a new data has arrived

            imuPackets = imuData.packets
            for imuPacket in imuPackets:
                acceleroValues = imuPacket.acceleroMeter
                gyroValues = imuPacket.gyroscope
                acceleroTs = acceleroValues.getTimestampDevice()
                gyroTs = gyroValues.getTimestampDevice()
                ts = acceleroTs if acceleroTs < gyroTs else gyroTs

                # print('time: {}'.format(ts.total_seconds()))
                ig.save(ts.total_seconds(), acceleroValues, gyroValues)

                timestamps[1:] = timestamps[:-1]
                timestamps[0] = ts.total_seconds()
                acc_data[1:, :] = acc_data[:-1, :]
                acc_data[0, :] = (acceleroValues.x, acceleroValues.y, acceleroValues.z)
                gyr_data[1:, :] = gyr_data[:-1, :]
                gyr_data[0, :] = (gyroValues.x, gyroValues.y, gyroValues.z)
                #
                # imuF = "{:.06f}"
                # tsF = "{:.03f}"
                #
                # print(f"Accelerometer timestamp: {tsF.format(acceleroTs)} ms")
                # print(
                #     f"Accelerometer [m/s^2]: x: {imuF.format(acceleroValues.x)} y: {imuF.format(acceleroValues.y)} z: {imuF.format(acceleroValues.z)}")
                # print(f"Gyroscope timestamp: {tsF.format(gyroTs)} ms")
                # print(
                #     f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)} ")

            if plot:
                graph1[0].remove()
                graph1 = ax.plot(timestamps, acc_data[:, 0], 'r', label='x')
                graph2[0].remove()
                graph2 = ax.plot(timestamps, acc_data[:, 1], 'b', label='y')
                graph3[0].remove()
                graph3 = ax.plot(timestamps, acc_data[:, 2], 'g', label='z')
                ax.legend()
                ax.set_xlim(timestamps[-1], timestamps[0])

                fig.canvas.draw()
                fig.canvas.flush_events()
                # TODO: add gyro data plot. for now, plot too slow!

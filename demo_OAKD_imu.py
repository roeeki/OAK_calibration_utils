#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import os
import datetime
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
import time
import cycler


if __name__ == "__main__":

    num_measurements_to_plot = 100

    # matplotlib.rcParams['interactive'] == True
    matplotlib.interactive(True)

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
    ax1 = fig.add_subplot(2, 1, 1)
    graph1 = ax1.plot(timestamps, acc_data[:, 0], 'r')
    graph2 = ax1.plot(timestamps, acc_data[:, 1], 'b')
    graph3 = ax1.plot(timestamps, acc_data[:, 2], 'g')
    ax1.set_xlabel("time [sec]")
    ax1.set_ylabel("acceleration [n/sec^2]")
    ax1.set_title('accelerometer data')

    ax2 = fig.add_subplot(2, 1, 2)
    graph11 = ax1.plot(timestamps, gyr_data[:, 0], 'r')
    graph22 = ax1.plot(timestamps, gyr_data[:, 1], 'b')
    graph33 = ax1.plot(timestamps, gyr_data[:, 2], 'g')
    ax2.set_xlabel("time [sec]")
    ax2.set_ylabel("angular velocity [rad/sec]")
    ax2.set_title('gyro data')
    # plt.ylim(-10, 10)

    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.0001)

    # Pipeline is defined, now we can connect to the device
    with dai.Device(pipeline) as device:

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

                timestamps[1:] = timestamps[:-1]
                timestamps[0] = ts.total_seconds()
                acc_data[1:, :] = acc_data[:-1, :]
                acc_data[0, :] = (acceleroValues.x, acceleroValues.y, acceleroValues.z)
                gyr_data[1:, :] = gyr_data[:-1, :]
                gyr_data[0, :] = (gyroValues.x, gyroValues.y, gyroValues.z)

            # TODO:make more efficient! clear and plot is slow!
            graph1[0].remove()
            graph2[0].remove()
            graph3[0].remove()
            graph1 = ax1.plot(timestamps, acc_data[:, 0], 'r', label='x')
            graph2 = ax1.plot(timestamps, acc_data[:, 1], 'b', label='y')
            graph3 = ax1.plot(timestamps, acc_data[:, 2], 'g', label='z')
            ax1.legend()
            ax1.set_xlim(timestamps[-1], timestamps[0])

            graph11[0].remove()
            graph22[0].remove()
            graph33[0].remove()
            graph11 = ax2.plot(timestamps, gyr_data[:, 0], 'r', label='x')
            graph22 = ax2.plot(timestamps, gyr_data[:, 1], 'b', label='y')
            graph33 = ax2.plot(timestamps, gyr_data[:, 2], 'g', label='z')
            ax2.legend()
            ax2.set_xlim(timestamps[-1], timestamps[0])

            fig.canvas.draw()
            fig.canvas.flush_events()

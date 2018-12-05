import csv
import datetime
import math
import os
import sys
import time
import timeit

import cv2
import numpy as np
import pyzed.camera as zcam
import pyzed.core as core
import pyzed.defines as sl
import pyzed.types as tp


def current_date_time():
    ts = time.time()
    current_date_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return current_date_time


def create_dir():
    # create dir based on current date time
    time_stamp = current_date_time()
    os.mkdir('./zed/' + time_stamp)
    dir_path = './zed/' + time_stamp
    return dir_path


directory_path = create_dir()
log_file_path = os.path.join(directory_path, current_date_time() + '.csv')


def main():
    # Create a PyZEDCamera object
    zed = zcam.PyZEDCamera()

    # Create a PyInitParameters object and set configuration parameters
    init_params = zcam.PyInitParameters()
    # These settings adjust the level of accuracy, range and computational performance of the depth sensing module. available modes are Ultra, quality, medium & performance
    init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_ULTRA  # Use ULTRA depth mode for better depth accuracy
    init_params.coordinate_units = sl.PyUNIT.PyUNIT_CENTIMETER  # Use milliliter units (for depth measurements)
    init_params.camera_fps = 15  # camera FPS
    init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD1080  # camera resolution

    # Open the camera
    err = zed.open(init_params)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)

    # Create and set PyRuntimeParameters after opening the camera
    runtime_parameters = zcam.PyRuntimeParameters()
    runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_FILL  # Use STANDARD sensing mode

    # Capture 50 images and depth, then stopdepth
    i = 0
    right_image = core.PyMat()
    left_image = core.PyMat()
    depth = core.PyMat()
    point_cloud = core.PyMat()

    # directory_path = create_dir()

    # for log file name time stamping

    while i < 50:
        # A new image is available if grab() returns PySUCCESS
        if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
            # Retrieve left image
            zed.retrieve_image(left_image, sl.PyVIEW.PyVIEW_LEFT)
            # Retrieve right image
            zed.retrieve_image(right_image, sl.PyVIEW.PyVIEW_RIGHT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            start = timeit.default_timer()
            zed.retrieve_measure(point_cloud, sl.PyMEASURE.PyMEASURE_XYZRGBA)
            stop = timeit.default_timer()

            print('Time took for pointcloud calculations: ', stop - start)

            # saving image on disk
            timestamp = zed.get_timestamp(
                sl.PyTIME_REFERENCE.PyTIME_REFERENCE_CURRENT)  # Get the timestamp at the time the image was captured
            left_image_path = timestamp.__str__() + '_L' + '.png'
            right_image_path = left_image_path.replace('_L.png', '') + '_R' + '.png'
            depth_image_path = left_image_path.replace('_L.png', '') + '_D' + '.png'

            cv2.imwrite(directory_path + '/' + left_image_path, left_image.get_data())
            cv2.imwrite(directory_path + '/' + right_image_path, right_image.get_data())
            cv2.imwrite(directory_path + '/' + depth_image_path, depth.get_data())

            # displaying image
            cv2.imshow('left image', left_image.get_data())
            cv2.imshow('depth image', depth.get_data())

            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            x = round(left_image.get_width() / 2)
            y = round(left_image.get_height() / 2)
            err, point_cloud_value = point_cloud.get_value(x, y)

            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                 point_cloud_value[1] * point_cloud_value[1] +
                                 point_cloud_value[2] * point_cloud_value[2])

            # forming list of values for image name, dumping depth, point cloud depth, x, y, z
            log_values = [left_image_path.__str__(), depth.get_value(x, y)[1], distance, point_cloud_value[0],
                          point_cloud_value[1],
                          point_cloud_value[2]]
            write_log(log_values, i)

            if not np.isnan(distance) and not np.isinf(distance):
                distance = round(distance)
                print("Distance to Camera at ({0}, {1}): {2} mm\n".format(x, y, distance))
                # Increment the loop
                i = i + 1
            else:
                print("Can't estimate distance at this position, move the camera\n")
            sys.stdout.flush()

    # Close the camera
    zed.close()


def write_log(log_values, i):
    f = open(log_file_path, 'a')
    writer = csv.writer(f, delimiter=',')
    if (i == 0):
        header = ['ImageName', 'DepthValue', 'PointCloudDepth', 'Xpoint', 'Ypoint', 'Zpoint']
        writer.writerow(header)
    writer.writerow(log_values)
    f.close()


if __name__ == "__main__":
    main()

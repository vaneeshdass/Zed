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


log_file_path = os.path.join(os.getcwd() + '/zed/', current_date_time() + '.csv')


def main():
    # Create a PyZEDCamera object
    zed = zcam.PyZEDCamera()

    # Create a PyInitParameters object and set configuration parameters
    init_params = zcam.PyInitParameters()
    init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
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
    image = core.PyMat()
    depth = core.PyMat()
    point_cloud = core.PyMat()

    # for log file name time stamping

    while i < 50:
        # A new image is available if grab() returns PySUCCESS
        if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.PyVIEW.PyVIEW_LEFT)
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
            image_path = timestamp.__str__() + '.png'
            depth_image_path = image_path.replace('.png', '') + '_D' + '.png'

            cv2.imwrite('./zed/' + image_path, image.get_data())
            cv2.imwrite('./zed/' + depth_image_path, depth.get_data())

            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            x = round(image.get_width() / 2)
            y = round(image.get_height() / 2)
            err, point_cloud_value = point_cloud.get_value(x, y)

            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                 point_cloud_value[1] * point_cloud_value[1] +
                                 point_cloud_value[2] * point_cloud_value[2])

            # forming list of values for image name, dumping depth, point cloud depth, x, y, z
            log_values = [image_path.__str__(), depth.get_value(x, y)[1], distance, point_cloud_value[0],
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
        header = ['Image Name', 'Depth Value', 'Point Cloud Depth', 'X point', 'Y point', 'Z point']
        writer.writerow(header)
    writer.writerow(log_values)
    f.close()


if __name__ == "__main__":
    main()

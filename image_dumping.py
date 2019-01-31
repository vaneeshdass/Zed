import datetime
import os
import time

import pyzed.camera as zcam
import pyzed.core as core
import pyzed.defines as sl
import pyzed.types as tp

from yolov3_detection import *


def current_date_time():
    ts = time.time()
    current_date_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return current_date_time


def create_dir():
    # create dir based on current date time
    time_stamp = current_date_time()
    if not os.path.exists('zed'):
        os.mkdir('./zed')
    os.mkdir('./zed/' + time_stamp)
    dir_path = './zed/' + time_stamp
    return dir_path


def main():
    # Create a PyZEDCamera object
    zed = zcam.PyZEDCamera()

    # Create a PyInitParameters object and set configuration parameters
    init_params = zcam.PyInitParameters()
    # These settings adjust the level of accuracy, range and computational performance of the depth sensing module.
    # available modes are Ultra, quality, medium & performance
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

    # Capture 50 images and depth, then stop depth
    right_image = core.PyMat()
    left_image = core.PyMat()
    depth = core.PyMat()
    point_cloud = core.PyMat()
    # this varaible we only used to display the depth view. its display in 8 bit
    depth_image_for_view_8_bit = core.PyMat()

    dump_image = False
    flip_image = False
    create_directory = True
    while True:
        # A new image is available if grab() returns PySUCCESS
        if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
            # Retrieve left image
            zed.retrieve_image(left_image, sl.PyVIEW.PyVIEW_LEFT)
            # Retrieve right image
            zed.retrieve_image(right_image, sl.PyVIEW.PyVIEW_RIGHT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.PyMEASURE.PyMEASURE_XYZRGBA)
            # Retrieve depth image for display only
            zed.retrieve_image(depth_image_for_view_8_bit, sl.PyVIEW.PyVIEW_DEPTH)

            # saving image on disk
            timestamp = zed.get_timestamp(
                sl.PyTIME_REFERENCE.PyTIME_REFERENCE_CURRENT)  # Get the timestamp at the time the image was captured
            left_image_path = current_date_time().__str__() + '.png'

            if flip_image:
                # flipping the image 180 degree(vertically)
                left_flipped_image_180 = cv2.rotate(left_image.get_data(), rotateCode=cv2.ROTATE_180)

            if dump_image:
                if create_directory:
                    directory_path = create_dir()
                    create_directory = False
                if flip_image:
                    cv2.imwrite(directory_path + '/' + left_image_path, left_flipped_image_180)
                else:
                    cv2.imwrite(directory_path + '/' + left_image_path, left_image.get_data())

            if flip_image:
                cv2.imshow('left image', left_flipped_image_180)

            else:
                cv2.imshow('left image', left_image.get_data())

            key = cv2.waitKey(1)

            if key == 27:  # if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break
            elif key == 32:  # if space is pressed start dumping the images
                dump_image = True
                print(
                    '------------------------------------------------------DUMPING IMAGES--------------------------------------------------------------')
            elif key == 120 or key == 88:  # if x is pressed stop dumping the images
                dump_image = False
                print(
                    '------------------------------------------------------STOP DUMPING IMAGES--------------------------------------------------------------')
            elif key == 102 or key == 70:  # if f is pressed then flipped the image
                flip_image = not flip_image

    # Close the camera
    zed.close()
    print(
        '------------------------------------------------------------program ends----------------------------------------------------------')


if __name__ == "__main__":
    main()

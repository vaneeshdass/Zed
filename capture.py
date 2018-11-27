import cv2
import pyzed.camera as zcam
import pyzed.core as core
import pyzed.defines as sl
import pyzed.types as tp


def main():
    # Create a PyZEDCamera object
    zed = zcam.PyZEDCamera()

    # Create a PyInitParameters object and set configuration parameters
    init_params = zcam.PyInitParameters()
    init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD1080  # Use HD1080 video mode
    init_params.camera_fps = 15  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)

    # Capture 50 frames and stop
    i = 0
    image = core.PyMat()
    runtime_parameters = zcam.PyRuntimeParameters()
    while i < 50:
        # Grab an image, a PyRuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
            # A new image is available if grab() returns PySUCCESS
            zed.retrieve_image(image, sl.PyVIEW.PyVIEW_LEFT)
            timestamp = zed.get_timestamp(
                sl.PyTIME_REFERENCE.PyTIME_REFERENCE_CURRENT)  # Get the timestamp at the time the image was captured
            cv2.imwrite('./' + timestamp.__str__() + '.png', image.get_data())
            print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
                                                                                 timestamp))
            i = i + 1

    # Close the camera
    zed.close()


if __name__ == "__main__":
    main()

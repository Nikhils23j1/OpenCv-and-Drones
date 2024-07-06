import cv2
import cv2.aruco as aruco
import numpy as np

def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

def markers(s,mtx,dist):
    # Load the input image
    img = cv2.imread(s)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the ArUco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

    # Initialize the detector parameters
    parameters = aruco.DetectorParameters_create()

    # Detect ArUco markers in the image
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw the bounding box for the detected marker
        img = aruco.drawDetectedMarkers(img, corners, ids)

        # Get the rotation and translation vectors for the detected marker
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

        # Compute the center of the marker
        xc, yc = np.mean(corners[0][0], axis=0)

        # Compute the pose axes for the detected marker
        axis_length = 0.1
        axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)

        for i in range(len(ids)):
            img = aruco.drawAxis(img, mtx, dist, rvec[i], tvec[i], axis_length)

        # Display the final output image
        cv2.imshow('ArUco Marker Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return xc, yc

    else:
        print("No ArUco marker detected in the given image.")
        return None

# Example usage:
loaded_coefficients = load_coefficients("calibration.yaml")
mtx, dist = loaded_coefficients
image_path = "img0.png"
markers(image_path,mtx,dist)

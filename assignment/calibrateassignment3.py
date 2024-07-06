import numpy as np
import cv2
import glob
import argparse


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



def markers(s, mtx, dist):
    # Load the image
    img = cv2.imread(s)

    # Define ArUco dictionary and parameters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()

    # Detect markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    # Check if any markers are detected
    if markerIds is not None and len(markerIds) > 0:
        # Draw bounding box for the detected marker
        cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)

        # Estimate pose of the detected marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.02, mtx, dist)

        # Draw pose axes for each detected marker
        for i in range(len(markerIds)):
            cv2.aruco.drawAxis(img, mtx, dist, rvecs[i], tvecs[i], 0.01)

        # Get translation and rotation vectors for the first marker (assuming only one marker is present)
        translation_vector = tvecs[0].flatten()
        rotation_vector = rvecs[0].flatten()

        # Compute center of the marker
        xc, yc = np.mean(markerCorners[0][0], axis=0)

        # Display the image with the bounding box and pose axes
        cv2.imshow('Detected Markers', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Return relevant information
        return xc, yc, translation_vector, rotation_vector
    else:
        print("No markers detected in the image.")
        return None



image_path = '/Users/nikhil/My Computer/A Year 2/DSA/python/image processing/Assignment 2/img0.png'
loaded_coefficients = load_coefficients("/Users/nikhil/My Computer/A Year 2/DSA/python/image processing/Assignment 2/calibration.yaml")
mtx, dist = loaded_coefficients
result = markers(image_path, mtx, dist)

if result is not None:
    xc, yc, translation_vector, rotation_vector = result
    print("Center of Marker (xc, yc):", xc, yc)
    print("Translation Vector:", translation_vector)
    print("Rotation Vector:", rotation_vector)

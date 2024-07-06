import cv2
# pressing  s key will save the image of your laptop video camera
cap=cv2.VideoCapture(0)
num=0
while True:
    ret,img=cap.read()
    k=cv2.waitKey(5)
    if k==27:
        break
    elif k==ord('s'):
        
        cv2.imwrite('img'+str(num)+'.png',img)
        print("image saved")
        num+=1
    cv2.imshow('img', img)
cap.release()
cv2.destroyAllWindows()


# import numpy as np
# import cv2 as cv
# import glob

# chessboard_size = (10,7)
# framesize = (640, 480)

# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# size_of_chessboard_squares_mm = 20
# objp = objp * size_of_chessboard_squares_mm
# objponts = []
# imgpoints = []

# imagepaths=['WhatsApp Image 2023-12-20 at 17.17.11.jpeg']

# for imagepath in imagepaths:
#     img = cv.imread(imagepath)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
#     if ret:
#         objponts.append(objp)
#         corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         imgpoints.append(corners2)
#         cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(500)
#     else:
#         print(f"Chessboard corners not found in {imagepath}")
# cv.waitKey(0)
# cv.destroyAllWindows()

# # Calibration

# ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objponts, imgpoints, framesize, None, None)
# print("Camera matrix: \n", cameraMatrix)




import cv2
import numpy as np
import matplotlib.pyplot as plt

def Sobel(s):
    img = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Edge-Detected Image')
    plt.axis('off')
    plt.show()


image_path = 'M.png' 
Sobel(image_path)

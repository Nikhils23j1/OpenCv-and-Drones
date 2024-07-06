import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def show(name, n,m,i, Title):
    plt.subplot(n,m,i)
    plt.imshow(name, cmap='gray')
    plt.title(Title)
    plt.axis('off')
def Canny(s):
    img = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)
    show(img,1,2,1,'Original')
    show(img,1,2,2,'Canny Edge Detection')
    plt.show()

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def series(img, is_lowpass, d):
    Img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (256, 256))
    show(Img, 3,5, 1 + 5 * is_lowpass, 'org')

    FFT = np.fft.fft2(Img)
    FFT_shifted = np.fft.fftshift(FFT)
    Scale = np.log1p(np.abs(FFT_shifted))
    show(Scale, 3,5, 2 + 5 * is_lowpass, 'fft_org')

    if is_lowpass:
        Filter = np.zeros((256, 256))
        Filter[(128 - d):(128 + d), (128 - d):(128 + d)] = 1
        show(Filter, 3,5, 3 + 5 * is_lowpass,  'filter')
    else:
        Filter = np.ones((256, 256))
        Filter[(128 - d):(128 + d), (128 - d):(128 + d)] = 0
        show(Filter, 3,5, 3 + 5 * is_lowpass, 'filter')

    G_shift = FFT_shifted * Filter
    G = np.fft.ifftshift(G_shift)
    show(np.log1p(np.abs(G_shift)), 3,5, 4 + 5 * is_lowpass, 'Convoluted')
    res = np.abs(np.fft.ifft2(G))
    show(res,3,5, 5 + 5 * is_lowpass, 'Final')
    return res

img1 = cv2.imread('/Users/nikhil/My Computer/A Year 2/DSA/python/image processing/assignment1/M.png')
img2 = cv2.imread('/Users/nikhil/My Computer/A Year 2/DSA/python/image processing/assignment1/F.png')

res1 = series(img1, 0, 10)
res2 = series(img2, 1, 10)

hybrid_img = (res1 + res2).clip(0, 255).astype(np.uint8)

show(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), 3,5, 11,  'Image 1')
show(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB),3,5,12, 'Image 2')
show(hybrid_img, 3,5,13, 'Hybrid Image')  # Fixed the variable name
plt.show()


# img=cv2.imread ("C:/Users/aravi/OneDrive/Desktop/tree.jpg")
# # cv2. imshow("Image',img)
# # cv2. waitKey (0)
# # cv2.destroyAllWindows ()
# plt.imshow(img[:,:,::-1]);plt.title("Original Image");plt.axis

# gray=cv2. cvtColor(img, cv2. COLOR_BGR2GRAY)
# plt.imshow(gray,
# cmap= 'gray');plt.title("Original Image");plt.axis("off");

# gray_inv=cv2.bitwise_not (gray)
# plt. imshow (gray_inv, cmap="gray");plt.title("Original Image");plt.axis("off");
# contour, hierarchy=cv2.findContours (gray_inv, cv2.RETR_EXTERNAL, cv2. CHAIN_APPROX_NONE)
# img_copy=img. copy()
# cv2 .drawContours (img_copy, contours, -1, (0,255, 0), 10)
#  plt.figure(figsize=[10,10])

# plt.imshow(img_copy[:,:,::-1]);plt.title("Original Image");plt.axis ("off");
# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
# Edge detection ( Canny edge implementation)
# from scipy import ndimage

# def sobel_filters(img):
#     Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
#     Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
#     Ix = ndimage.filters.convolve(img, Kx)
#     Iy = ndimage.filters.convolve(img, Ky)
    
#     G = np.hypot(Ix, Iy)
#     G = G / G.max() * 255
#     theta = np.arctan2(Iy, Ix)
    
#     return (G, theta)
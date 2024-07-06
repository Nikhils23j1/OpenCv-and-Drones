import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(name, n, m, i, Title):
    plt.subplot(n, m, i)
    plt.imshow(name, cmap='gray')
    plt.title(Title)
    plt.axis('off')

def colour(s, target_color):
    # Read the input image
    image = cv2.imread(s)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    show(image_rgb,1,2,1,'Original Image')
    

    # Define color thresholds based on the target color
    if target_color.lower() == 'red':
        lower_threshold = np.array([100, 0, 0], dtype=np.uint8)
        upper_threshold = np.array([255, 80, 80], dtype=np.uint8)
    elif target_color.lower() == 'green':
        lower_threshold = np.array([0, 100, 0], dtype=np.uint8)
        upper_threshold = np.array([80, 255, 80], dtype=np.uint8)
    elif target_color.lower() == 'blue':
        lower_threshold = np.array([0, 0, 100], dtype=np.uint8)
        upper_threshold = np.array([80, 80, 255], dtype=np.uint8)
    else:
        raise ValueError("Unsupported target color")

    # Create a binary mask based on the color thresholds
    mask = cv2.inRange(image_rgb, lower_threshold, upper_threshold)
    mask2 = cv2.inRange(image, lower_threshold, upper_threshold)
    
    # Create an inverted binary mask of the original image_rgb
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    # show(gray,1,2,2,'Binary Image')
    _, binary = cv2.threshold(gray_inv, 50, 255, cv2.THRESH_BINARY)
    
    contours, heirarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("Number of contours found = {}".format(len(contours)))
    print(heirarchy)
    
    # Find contours in the binary mask
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("Number of contours found = {}".format(len(contours)))
    print(heirarchy)
    
    result_image = image_rgb.copy()
    cv2.drawContours(result_image, contours, -1, (0, 0,0), 4)

    # Convert the result image back to BGR

    # Display the original and result images
    show(result_image,1,2,2,'Contours on Original Image')
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Save the result image to a file
    # cv2.imwrite('result_image.png', result_image_bgr)

# Example usage:
# imagepath='/Users/nikhil/My Computer/A Year 2/DSA/python/image processing/Midterm/color-shapes.png'
imagepath='/Users/nikhil/My Computer/A Year 2/DSA/python/image processing/Midterm/PIC-s-s-c.png'
colour(imagepath, target_color='red')

import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(name, n, m, i, Title):
    plt.subplot(n, m, i)
    plt.imshow(name, cmap='gray')
    plt.title(Title)
    plt.axis('off')

def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

img = cv2.imread('/Users/nikhil/My Computer/A Year 2/DSA/python/image processing/Midterm/types-of-polygons1.png')
sharpened_img = sharpen_image(img)

blurred = cv2.GaussianBlur(sharpened_img, (5, 5), 0)
edges = cv2.Canny(blurred, 100, 160)
show(edges, 1, 4, 1, 'Edges')

gray = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2GRAY)
gray_inv = cv2.bitwise_not(gray)
_, binary = cv2.threshold(gray_inv, 50, 255, cv2.THRESH_BINARY)
show(binary, 1, 4, 2, 'Binary Image')



contours, heir = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key = cv2.contourArea)
cv2.drawContours(binary,[c],0,(0, 255, 0),3)
show(binary[:, ::-1], 1, 4, 3, "Original Image")

plt.show()



import cv2
import numpy as np
import matplotlib.pyplot as plt


import cv2  # Import the OpenCV library
import matplotlib.pyplot as plt  # Import the Matplotlib library for visualization

img = cv2.imread("C:/Users/aravi/OneDrive/Desktop/image.png")  # Read the original image
plt.imshow(img[:, :, ::-1])  # Display the original image in Matplotlib
#The ::-1 part reverses the order of elements along the third axis, which corresponds to the color channels (typically red, green, and blue in RGB images).
#So, img[:, :, ::-1] essentially means to take all the rows and columns of the image and reverse the order of the color channels. This is commonly used when displaying images with Matplotlib, as Matplotlib expects the color channels in the order blue, green, and red (BGR), whereas OpenCV (used for image reading) represents images in the order red, green, and blue (RGB).
plt.title("Original Image")  # Set the title for the image
plt.axis("off")  # Turn off axis labels

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
plt.imshow(gray, cmap='gray')  # Display the grayscale image
plt.title("Grayscale Image")  # Set the title for the image
plt.axis("off")  # Turn off axis labels

gray_inv = cv2.bitwise_not(gray)  # Create the inverted grayscale image
plt.imshow(gray_inv, cmap="gray")  # Display the inverted grayscale image
plt.title("Inverted Grayscale Image")  # Set the title for the image
plt.axis("off")  # Turn off axis labels

contours, hierarchy = cv2.findContours(gray_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours in the inverted grayscale image
img_copy = img.copy()  # Create a copy of the original image


cv2.drawContours(img_copy, contours, -1, (255, 0, 0), 2)  # Draw contours on the copied image

plt.figure(figsize=[10, 10])  # Set the size of the Matplotlib figure
plt.imshow(img_copy[:, :, ::-1])  # Display the original image with contours
plt.title("Original Image with Contours")  # Set the title for the image
plt.axis("off")  # Turn off axis labels

_, binary = cv2.threshold(gray_inv, 25, 255, cv2.THRESH_BINARY)  # Create a binary image
plt.imshow(binary, cmap="gray")  # Display the binary image
plt.title("Binary Image")  # Set the title for the image
plt.axis("off")  # Turn off axis labels

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours in the binary image
img_copy = img.copy()  # Create a copy of the original image
cv2.drawContours(img_copy, contours, -1, (255, 0, 0), 2)  # Draw contours on the copied image

plt.figure(figsize=[8, 8])  # Set the size of the Matplotlib figure
plt.imshow(img_copy[:, :, ::-1])  # Display the original image with contours
plt.title("Original Image with Contours")  # Set the title for the image
plt.axis("off")  # Turn off axis labels

img = cv2.imread("C:/Users/aravi/OneDrive/Desktop/sword.jpg")  # Read another original image
plt.figure(figsize=[10, 10])  # Set the size of the Matplotlib figure
plt.imshow(img[:, :, ::-1])  # Display the other original image
plt.title("Original Image")  # Set the title for the image
plt.axis("off")  # Turn off axis labels

blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian blur to the image
edges = cv2.Canny(blurred, 100, 160)  # Detect edges in the blurred image

plt.figure(figsize=[10, 10])  # Set the size of the Matplotlib figure
plt.imshow(edges, cmap="gray")  # Display the edges image
plt.title("Edges Image")  # Set the title for the image

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the edges image
img_copy = img.copy()  # Create a copy of the original image
cv2.drawContours(img_copy, contours, -1, (255, 0, 0), 2)  # Draw contours on the copied image

plt.figure(figsize=[10, 10])  # Set the size of the Matplotlib figure
plt.imshow(img_copy[:, :, ::-1])  # Display the original image with contours
plt.title("Original Image with Contours")  # Set the title for the image
plt.axis("off")  # Turn off axis labels

plt.show()  # Show the Matplotlib plot

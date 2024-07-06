import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
# image_path = '/Users/nikhil/My Computer/A Year 2/DSA/python/image processing/Midterm/kl-h-building-d.jpg'
image_path='lines.jpg'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (5,5), 5)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened
gray_image = sharpen_image(gray_image)
# Apply edge detection (optional, but often helpful)
edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

# Apply Hough Lines Transform
lines = cv2.HoughLines(edges,rho= 1, theta= np.pi / 180, threshold=100)

# Draw the lines on the original image
for line in lines:
	rho, theta = line[0]
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a * rho
	y0 = b * rho
	x1 = int(x0 + 1000 * (-b))
	y1 = int(y0 + 1000 * (a))
	x2 = int(x0 - 1000 * (-b))
	y2 = int(y0 - 1000 * (a))
	cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255),2)

# Display the result
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image with Hough Lines'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()







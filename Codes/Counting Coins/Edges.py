import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image and Convert to grayscale
image = cv2.imread('images/one.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')  
plt.show()

# 2. Apply Gaussian smoothing to reduce noise
blur = cv2.GaussianBlur(gray, (11,11), 0)
plt.imshow(blur, cmap='gray')
plt.title('Smoothed Grayscale Image')
plt.axis('off')  
plt.show()

# 3. Morphological Opening to remove small noise
kernel = np.ones((15,15), np.uint8)
opened = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, iterations=2)

# 4. Edge Detection
canny = cv2.Canny(opened, 50, 90)  
plt.imshow(canny, cmap='gray')
plt.title('Edge Detection with Text Reduction')
plt.axis('off')
plt.show()




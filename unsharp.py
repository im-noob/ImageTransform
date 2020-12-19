

import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
import numpy as np
import cv2
import PIL.Image, PIL.ImageTk
original_image = plt.imread('per.jpeg').astype('uint16')

# Convert to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Median filtering
gray_image_mf = median_filter(gray_image, 1)

# Calculate the Laplacian
lap = cv2.Laplacian(gray_image_mf,cv2.CV_64F)

# Calculate the sharpened image
sharp = gray_image - 0.9*lap
# img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(sharp))
# plt.show()
cv2.imshow('image',sharp)
while True:
	k = cv2.waitKey(100) 
	# change the value from the original 0 (wait forever) to something appropriate
	if k == 27:
		print('ESC')
		cv2.destroyAllWindows()
		break        
	if cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:        
		break        
# cv2.destroyAllWindows()


# cv2.imshow("test", sharp)
# k = cv2.waitKey(0) & 0xFF

import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt
'''
def colourThreshold():
    img = cv2.imread('../testImages/LeptonImages/A2_Monday13May_fullLecutre.png')

    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(img, lower_red, upper_red)
 
    # Range for upper range
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(img,lower_red,upper_red)
 
    # Generating the final mask to detect red color
    mask1 = mask1+mask2
'''

# Capture the mouse click events in Python and OpenCV
'''
-> draw shape on any image 
-> reset shape on selection
-> crop the selection
run the code : python capture_events.py --image image_example.jpg
'''


# import the necessary packages
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
ref_point = []
cropping = False

def shape_selection(event, x, y, flags, param):
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		print(x, y)
		#BGRcolour = image[x, y]
		#HSVcolour = HSVimage[x, y]
		#print("BGR: ");
		#print(BGRcolour[0], BGRcolour[1], BGRcolour[2]) # b, g, r
		#print("HSV: ");
		#print(HSVcolour[0], HSVcolour[1], HSVcolour[2]) # 
		cv2.imshow("image", image)

# load the image, clone it, and setup the mouse callback function
image = cv2.imread('../testImages/LeptonImages/A2_Monday13May_fullLecutre.png')
HSVimage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", shape_selection)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

# close all open windows
cv2.destroyAllWindows()

    
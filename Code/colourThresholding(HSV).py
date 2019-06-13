import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

def threshold():
    image = cv2.imread('../testImages/LeptonImages/A2_Monday13May_fullLecutre.png')
    HSVimage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    #cv2.namedWindow('Threshold Transform', cv2.WINDOW_AUTOSIZE)
    #cv2.createTrackbar('Red Threshold', 'Threshold Transform', 0, 255, nothing)
    #cv2.createTrackbar('Hue Threshold', 'Threshold Transform', 0, 255, nothing)
    
    clone = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #image.copy()
    HSVclone = cv2.cvtColor(HSVimage, cv2.COLOR_BGR2GRAY)# HSVimage.copy()
    
    cv2.imshow('grey image', clone)
    cv2.imshow('grey hsv image', HSVclone)    
    
    while True:
	#redThreshold = cv2.getTrackbarPos('Red Threshold', 'Threshold Transform')
	#hueThreshold = cv2.getTrackbarPos('Hue Threshold', 'Threshold Transform')
	
	'''
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]): # b g e
                if(image[i, j][2] > redThreshold):
                    clone[i][j] = 1
                else:
                    clone[i][j] = 0
		
		
                if(HSVimage[i, j][2] > hueThreshold):
                    HSVclone[i][j] = 1
                else:
                    HSVclone[i][j] = 0   
		#print('HSV image value: ')
		#print(HSVimage[i, j][2])
	'''	
		
	#combined = np.concatenate((clone, HSVclone), axis=1)
	#cv2.imshow('Threshold Transform', combined)
	#cv2.imshow('image', image)
	#cv2.imshow('HSVimage', HSVimage)
	key = cv2.waitKey(10) & 0xFF
	if key == ord("c"):
		break        
	    


if __name__ == "__main__":
    threshold()
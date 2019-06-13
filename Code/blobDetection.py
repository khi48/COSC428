import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

def threshold():
    img = cv2.imread('../testImages/LeptonImages/A2_Monday13May_fullLecutre.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Blob Detection', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Threshold value', 'Blob Detection', 0, 255, nothing)
    cv2.createTrackbar('Erosion kernel size', 'Blob Detection', 1, 100, nothing)
    cv2.createTrackbar('Erosion iterations', 'Blob Detection', 1, 100, nothing)  
    cv2.createTrackbar('Opening kernel size', 'Blob Detection', 1, 100, nothing)
    
    while True:
        thresholdValue = cv2.getTrackbarPos('Threshold value', 'Blob Detection')
        erosionKernelSize = cv2.getTrackbarPos('Erosion kernel size', 'Blob Detection')
        erosionIterations = cv2.getTrackbarPos('Erosion iterations', 'Blob Detection')
        openingKernelSize = cv2.getTrackbarPos('Opening kernel size', 'Blob Detection')
        #gausKernelSize = cv2.getTrackbarPos('Gaussian Kernel Value', 'Blob Detection')//2*2+1
        #gausIterations = cv2.getTrackbarPos('Smoothing Iterations', 'Blob Detection')
        
        #print(erosionKernelSize)
        # thresholding
        ret,thresh1 = cv2.threshold(gray,thresholdValue,255,cv2.THRESH_BINARY_INV)               
        # morphology
        erosionKernel = np.ones((erosionKernelSize,erosionKernelSize),np.uint8)
        erosion = cv2.erode(thresh1, erosionKernel, iterations=erosionIterations)
        
        openingKernel = np.ones((openingKernelSize,openingKernelSize),np.uint8)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, openingKernel)
        
        #blob detection
        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create()
         
        # Detect blobs.
        keypoints = detector.detect(opening)
         
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(opening, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('Blob Detection', im_with_keypoints)     
        cv2.imshow('Origional', img)     

        if cv2.waitKey(100) & 0xFF == ord('q'):
            height, width = img.shape[:2]
            print('height: ' + str(height))            
            print('width: ' + str(width))
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    threshold()
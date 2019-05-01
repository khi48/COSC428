import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

def morphology():
    img = cv2.imread('../testImages/FLIRImages/thermal3.jpg')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Threshold Transform', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Threshold value', 'Threshold Transform', 0, 255, nothing)
    cv2.createTrackbar('Kernel Size', 'Threshold Transform', 1, 100, nothing)
    cv2.createTrackbar('Iterations', 'Threshold Transform', 1, 100, nothing)
    
    while True:
        thresholdValue = cv2.getTrackbarPos('Threshold value', 'Threshold Transform')
        kernelSize = cv2.getTrackbarPos('Kernel Size', 'Threshold Transform')
        morphIterations = cv2.getTrackbarPos('Iterations', 'Threshold Transform')
        
        ret,binaryThresh = cv2.threshold(gray,thresholdValue,255,cv2.THRESH_BINARY_INV)
        # morphological features
        kernel = np.ones((kernelSize,kernelSize),np.uint8)
        
        erosion = cv2.erode(binaryThresh, kernel, iterations=morphIterations)
        dilation = cv2.dilate(binaryThresh, kernel, iterations=morphIterations)
        opening = cv2.morphologyEx(binaryThresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(binaryThresh, cv2.MORPH_CLOSE, kernel)
        
        cv2.imshow('Threshold', binaryThresh)
        cv2.imshow('erosion', erosion)
        cv2.imshow('dilation', dilation)
        cv2.imshow('opening', opening)
        cv2.imshow('closing', closing)
        
        
        # Displaying Image
        width = 243 
        height = int(width*(img.shape[0]/img.shape[1]))
        dim = (width, height)

        binaryThresh = cv2.resize(binaryThresh, dim)
        erosion = cv2.resize(erosion, dim)
        dilation = cv2.resize(dilation, dim)
        opening = cv2.resize(opening, dim)
        closing = cv2.resize(closing, dim)
        
        combined = np.concatenate((binaryThresh, erosion, dilation, opening, closing), axis=1)
        cv2.imshow('Threshold Transform', combined)         
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    morphology()        
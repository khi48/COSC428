import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

'''
splitting image vertically. 0 is top
'''
def cropImage(image, minHeight, maxHeight): 
    croppedImg = image[minHeight:maxHeight]
    return croppedImg

def reconstructImage(reconstructedImg, croppedThresh, minHeight, maxHeight):
    reconstructedImg[minHeight:maxHeight] = croppedThresh
    return reconstructedImg
    
def splitImg():
    cv2.namedWindow('Image Split', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Threshold Constant', 'Image Split', 0, 255, nothing)
    cv2.createTrackbar('Threshold Variable', 'Image Split', 0, 255, nothing)  
    
    img = cv2.imread('../testImages/FLIRImages/thermal3.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape[:2]
    #print('height: ' + str(height))            
    #print('width: ' + str(width))    
    reconstructedImg = np.zeros((height, width), np.uint8) # possible that this needs to be copies some otherway
    
    while True:
        threshConstant = cv2.getTrackbarPos('Threshold Constant', 'Image Split')
        threshVar = cv2.getTrackbarPos('Threshold Variable', 'Image Split')
        
        numSplits = 10
        deltaHeight = int(height/numSplits)
        
        for i in range(0, numSplits):
            minHeight = int(i*deltaHeight)
            maxHeight = int((i+1)*deltaHeight)  
            
            croppedImg = cropImage(gray, minHeight, maxHeight)
            
            thresholdValue = threshConstant + threshVar*i
            ret,cropThresh = cv2.threshold(croppedImg,thresholdValue,255,cv2.THRESH_BINARY_INV)        
            
            reconstructedImg = reconstructImage(reconstructedImg, cropThresh, minHeight, maxHeight)
            
            cv2.imshow('halved %d'%i, cropThresh)
            
        cv2.imshow('Image Split', reconstructedImg)
        cv2.imshow('unaltered', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
if __name__ == "__main__":
    splitImg()    
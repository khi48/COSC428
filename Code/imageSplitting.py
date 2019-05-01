import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

'''
splitting image vertically
'''
def cropImage(image, i, deltaHeight):
    minHeight = int(i*deltaHeight)
    maxHeight = int((i+1)*deltaHeight)    
    croppedImg = image[minHeight:maxHeight]
    return croppedImg

def reconstructImage(reconstructedImg, croppedThresh, i, deltaHeight):
    minHeight = int(i*deltaHeight)
    maxHeight = int((i+1)*deltaHeight)    
    reconstructedImg[minHeight:maxHeight] = croppedThresh
    return reconstructedImg
    
def splitImg():
    cv2.namedWindow('Image Split', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Threshold Constant', 'Image Split', 0, 255, nothing)
    cv2.createTrackbar('Threshold Variable', 'Image Split', 0, 255, nothing)  
    
    img = cv2.imread('../testImages/FLIRImages/thermal3.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    reconstructedImg = np.zeros((height, width), np.uint8) # possible that this needs to be copies some otherway
    
    while True:
        threshConstant = cv2.getTrackbarPos('Threshold Constant', 'Image Split')
        threshVar = cv2.getTrackbarPos('Threshold Variable', 'Image Split')
        
        
        #print('height: ' + str(height))            
        #print('width: ' + str(width))
        #croppedImg = img[0:int(height/2)]
        numOfSplits = 10
        deltaHeight = int(height/numOfSplits)
        
        for i in range(0, numOfSplits):
            minHeight = int(i*deltaHeight)
            maxHeight = int((i+1)*deltaHeight)  
            
            croppedImg = cropImage(gray, i, deltaHeight)
            
            thresholdValue = threshConstant + threshVar*i
            ret,cropThresh = cv2.threshold(croppedImg,thresholdValue,255,cv2.THRESH_BINARY_INV)       
            ret,grayThresh = cv2.threshold(gray,thresholdValue,255,cv2.THRESH_BINARY_INV) 
            
            
            #reconstructedImg = reconstructImage(reconstructedImg, cropThresh, i, deltaHeight)
            cv2.imshow('halved %d'%i, cropThresh)
            cv2.waitKey(10)
            reconstructedImg[minHeight:maxHeight] = cropThresh
            #cv2.imshow('gray %d'%i, grayThresh) 
            
        cv2.imshow('Image Split', reconstructedImg)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
if __name__ == "__main__":
    splitImg()    
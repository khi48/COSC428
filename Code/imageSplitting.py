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
    
def splitImg():
    cv2.namedWindow('Image Split', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Threshold Constant', 'Image Split', 0, 255, nothing)
    cv2.createTrackbar('Threshold Variable', 'Image Split', 0, 255, nothing)  
    
    img = cv2.imread('../testImages/FLIRImages/thermal3.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    while True:
        threshConstant = cv2.getTrackbarPos('Threshold Constant', 'Image Split')
        threshVar = cv2.getTrackbarPos('Threshold Variable', 'Image Split')
        
        #print(cv2.Size(img))
        height, width = img.shape[:2]
        print('height: ' + str(height))            
        print('width: ' + str(width))
        croppedImg = img[0:int(height/2)]
        numOfSplits = 10
        deltaHeight = int(height/numOfSplits)
        for i in range(0, numOfSplits):
            cropped = cropImage(gray, i, deltaHeight)
            thresholdValue = threshConstant + threshVar*i
            ret,cropThresh = cv2.threshold(cropped,thresholdValue,255,cv2.THRESH_BINARY_INV)       
            ret,grayThresh = cv2.threshold(cropped,thresholdValue,255,cv2.THRESH_BINARY_INV) 
            
            cv2.imshow(str("halved %d"%i), cropThresh)
            cv2.waitKey(10)
            cv2.imshow('Image Split', grayThresh) 
            
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
if __name__ == "__main__":
    splitImg()    
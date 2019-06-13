import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):
    pass

def cropImage(image, minHeight, maxHeight): 
    croppedImg = image[minHeight:maxHeight]
    return croppedImg

def reconstructImage(reconstructedImg, croppedThresh, minHeight, maxHeight):
    reconstructedImg[minHeight:maxHeight] = croppedThresh
    return reconstructedImg


def threshold():
    
    image = cv2.imread('../testImages/LeptonImages/A2_Monday13May_fullLecutre.png')
    cv2.imshow('image',image)
    
    cv2.namedWindow('Threshold', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('constant', 'Threshold', 0, 255, nothing)
    cv2.createTrackbar('var', 'Threshold', 0, 255, nothing)
    cv2.createTrackbar('split#', 'Threshold', 1, 255, nothing)
    
    cv2.namedWindow('erosion', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('erosion K size', 'erosion', 1, 100, nothing)
    cv2.createTrackbar('erosion iterations', 'erosion', 1, 100, nothing)
    
    cv2.namedWindow('dilation', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('dilation K size', 'dilation', 1, 100, nothing)
    cv2.createTrackbar('dilation iterations', 'dilation', 1, 100, nothing)    
    
    cv2.namedWindow('closing', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('closing K size', 'closing', 1, 100, nothing)    

    cv2.namedWindow('BackgroundThresh', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('threshVal', 'BackgroundThresh', 1, 255, nothing)     
    
    image = cv2.imread('../testImages/LeptonImages/A2_Monday13May_fullLecutre.png')
    height, width = image.shape[:2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reconstructedImage = gray.copy()
    grayGaus = cv2.GaussianBlur(gray, (3, 3), 0) # kernel has to be odd
    
    HSVimage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    huh = HSVimage[:,:,2]
    HSVgray = cv2.cvtColor(HSVimage, cv2.COLOR_BGR2GRAY)
    cv2.imshow('HSVgray',HSVgray)
    HSVgrayGaus = cv2.GaussianBlur(HSVgray, (3, 3), 0) # kernel has to be odd  Probably remove this..    

    kernelSize = 1
    dilationIterations = 1 
    kernel = np.ones((kernelSize,kernelSize),np.uint8)
    
    HSVgrayThresholded = cv2.threshold(HSVgray,76,255,cv2.THRESH_BINARY)[1]
    cv2.imshow('HSVgrayThresholded before edit',HSVgrayThresholded)
    dilation = cv2.dilate(HSVgrayThresholded, kernel, iterations=1) # expands white object on black background
    cv2.imshow('dilation',dilation)
    erosion = cv2.erode(dilation, kernel, iterations=1) # reduces white object on black background
    cv2.imshow('erosion',erosion)
    goodParts = cv2.bitwise_and(huh,erosion)    
    cv2.imshow('HSVimage',HSVimage)
    cv2.imshow('huh',huh)
    
    
    
    cv2.imshow('gray',gray)
   
    
    cv2.imshow('HSVgrayGaus',HSVgrayGaus)
    while True:
        thresholdValue = cv2.getTrackbarPos('threshVal', 'BackgroundThresh')
        print('thresholdValue: %d'%thresholdValue)
        thresholdValue = 76
        
        numSplits = cv2.getTrackbarPos('split#', 'Threshold')
        diffThreshConstant = cv2.getTrackbarPos('constant', 'Threshold')
        diffThreshVar = cv2.getTrackbarPos('var', 'Threshold')  
        
        erosionKernelSize = cv2.getTrackbarPos('erosion K size', 'erosion')
        erosionIterations = cv2.getTrackbarPos('erosion iterations', 'erosion')
        
        dilationKernelSize = cv2.getTrackbarPos('dilation K size', 'dilation')
        dilationIterations = cv2.getTrackbarPos('dilation iterations', 'dilation')        
        
        closingKernelSize = cv2.getTrackbarPos('closing K size', 'closing')        
        #numSplits = 100
        #diffThreshConstant = 128 # removes centre front person. Doesnt filter out background at bottom. Pretty good otherwise 
        #diffThreshVar = 1  
        '''
        numSplits = 100
        diffThreshConstant = 128
        diffThreshVar = 1     
        '''
        #cv2.imshow('HSVgray',HSVgray)
        #ret,HSVgrayThresholded = cv2.threshold(HSVgray,thresholdValue,255,cv2.THRESH_BINARY)
        #cv2.threshold(HSVgray,thresholdValue,255,cv2.THRESH_BINARY)
        
        #HSVgrayThresholdedInv = cv2.bitwise_not(HSVgrayThresholded)
        #cv2.imshow('HSVgrayThresholded',HSVgrayThresholded)
        #cv2.imshow('BackgroundThresh',HSVgrayThresholdedInv)
        
        #goodParts = cv2.bitwise_and(huh,HSVgrayThresholded)
        cv2.imshow('goodParts',goodParts)
        outsuThresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        deltaHeight = float(height)/numSplits
        print('diffThreshConstant: %d, diffThreshVar: %.2f, height: %d, deltaHeight: %.2f'%(diffThreshConstant,diffThreshVar,height,deltaHeight))
        for i in range(0, numSplits+1):
            minHeight = int(i*deltaHeight)
            maxHeight = int((i+1)*deltaHeight) 
            croppedImg = cropImage(goodParts, minHeight, maxHeight)
            diffThreshValue = diffThreshConstant + diffThreshVar*(numSplits-i)
            ret,cropThreshImage = cv2.threshold(croppedImg,diffThreshValue,255,cv2.THRESH_BINARY)     
            reconstructedImage = reconstructImage(reconstructedImage, cropThreshImage, minHeight, maxHeight)
            
        diffThreshImage = reconstructedImage.copy()
        cv2.imshow('Threshold',diffThreshImage)
        
        erosionKernel = np.ones((erosionKernelSize,erosionKernelSize),np.uint8)
        dilationKernel = np.ones((dilationKernelSize,dilationKernelSize),np.uint8)
        closingKernel = np.ones((closingKernelSize,closingKernelSize),np.uint8)
        
        dilation = cv2.dilate(diffThreshImage, dilationKernel, iterations=dilationIterations) # expands white object on black background
        erosion = cv2.erode(dilation, erosionKernel, iterations=erosionIterations) # reduces white object on black background
        #opening = cv2.morphologyEx(diffThreshImage, cv2.MORPH_OPEN, openingKernel) # expands any gaps in white objects
        closing = cv2.morphologyEx(diffThreshImage, cv2.MORPH_CLOSE, closingKernel) # reduces any gaps in white objects 
        
        cv2.imshow('dilation',dilation)
        cv2.imshow('erosion',erosion)
        #cv2.imshow('opening',opening)
        cv2.imshow('closing',closing)
        cv2.waitKey(100)
    
if __name__ == "__main__":
    threshold()
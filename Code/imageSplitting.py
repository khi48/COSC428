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
    cv2.createTrackbar('Dilation kernel size', 'Image Split', 1, 100, nothing)
    cv2.createTrackbar('Dilation iterations', 'Image Split', 1, 100, nothing)  
    cv2.createTrackbar('Opening kernel size', 'Image Split', 1, 100, nothing)
    
    img = cv2.imread('../testImages/LeptonImages/A2_Monday13May_fullLecutre.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape[:2]
    #print('height: ' + str(height))            
    #print('width: ' + str(width))       
    
    reconstructedImg = np.zeros((height, width), np.uint8) # possible that this needs to be copies some otherway    
    
    # removing FLIR symbol 
    rectHeight = 65
    rectLength = 135
    gray[0:rectHeight,0:rectLength] = np.zeros((rectHeight, rectLength), np.uint8)
    
    while True:
        threshConstant = cv2.getTrackbarPos('Threshold Constant', 'Image Split')
        threshVar = cv2.getTrackbarPos('Threshold Variable', 'Image Split')
        #erosionKernelSize = cv2.getTrackbarPos('Erosion kernel size', 'Image Split')
        #erosionIterations = cv2.getTrackbarPos('Erosion iterations', 'Image Split')
        dilationKernelSize = cv2.getTrackbarPos('Dilation kernel size', 'Image Split')
        dilationIterations = cv2.getTrackbarPos('Dilation iterations', 'Image Split')        
        openingKernelSize = cv2.getTrackbarPos('Opening kernel size', 'Image Split')
        
        numSplits = 10
        deltaHeight = int(height/numSplits)
        
        for i in range(0, numSplits):
            minHeight = int(i*deltaHeight)
            maxHeight = int((i+1)*deltaHeight)  
            
            croppedImg = cropImage(gray, minHeight, maxHeight)
            
            thresholdValue = threshConstant + threshVar*i
            
            ret,cropThresh = cv2.threshold(croppedImg,thresholdValue,255,cv2.THRESH_BINARY)        
            
            reconstructedImg = reconstructImage(reconstructedImg, cropThresh, minHeight, maxHeight)
            
            #cv2.imshow('halved %d'%i, cropThresh)
            
        cv2.imshow('Image Split', reconstructedImg)
        
        #reconstructedImg = cv2.adaptiveThreshold(reconstructedImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY_INV, 11, 1) 
        #cv2.imshow('Image Split smoothed', reconstructedImg)
        
        #erosionKernel = np.ones((erosionKernelSize,erosionKernelSize),np.uint8)
        #erosion = cv2.erode(reconstructedImg, erosionKernel, iterations=erosionIterations)
        
        # Perform a simple blob detect
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 20  # The dot in 20pt font has area of about 30
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        detector = cv2.SimpleBlobDetector_create(params)
        
        dilationKernel = np.ones((dilationKernelSize,dilationKernelSize),np.uint8)
        dilation = cv2.dilate(reconstructedImg, dilationKernel, iterations=dilationIterations)        
        
        openingKernel = np.ones((openingKernelSize,openingKernelSize),np.uint8)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, openingKernel)
        
        # Blob detection
        # Set up the detector with default parameters.
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0 
        params.minThreshold = 0
        params.maxThreshold = 20
        params.filterByArea = True
        params.minArea = 20
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        
        detector = cv2.SimpleBlobDetector_create(params)
         
        # Detect blobs.
        keyPoints = detector.detect(opening)
        for keyPoint in keyPoints:
            x = keyPoint.pt[0]
            y = keyPoint.pt[1]
            s = keyPoint.size
            print("center y coordinate: " + str(y))
        print("------------------------------")
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(opening, keyPoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('Blob Detection', im_with_keypoints)     
        cv2.imshow('gray', gray)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
if __name__ == "__main__":
    splitImg()    
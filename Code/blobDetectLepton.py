''' 
Ideas to test:
    - Measure size of blob against the y co-ord and if large and far away, then aggresively threshold.
    - Check the size of the blob against the y co-ord and if small and close, the unthreshold.
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random as rng
import math


def nothing(x):
    pass

def cropImage(image, minHeight, maxHeight): 
    croppedImg = image[minHeight:maxHeight]
    return croppedImg

def reconstructImage(reconstructedImg, croppedThresh, minHeight, maxHeight):
    reconstructedImg[minHeight:maxHeight] = croppedThresh
    return reconstructedImg

def blobDetect():
    # reading origional image and creating greyscale images
    image = cv2.imread('../testImages/LeptonImages/A2_Monday13May_fullLecutre.png')
    gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    HSVimage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(HSVimage, cv2.COLOR_BGR2GRAY)
    reconstructedImage = gray.copy()
    height, width = gray.shape[:2]
    
    # creating track bars
    cv2.namedWindow('Threshold', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Threshold value', 'Threshold', 0, 255, nothing)
    
    cv2.namedWindow('erosion', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('erosion K size', 'erosion', 1, 100, nothing)
    cv2.createTrackbar('erosion iterations', 'erosion', 1, 100, nothing)
    
    cv2.namedWindow('dilation', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('dilation K size', 'dilation', 1, 100, nothing)
    cv2.createTrackbar('dilation iterations', 'dilation', 1, 100, nothing)
    
    cv2.namedWindow('diff threshold', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('number image splits', 'diff threshold', 1, 100, nothing)
    cv2.createTrackbar('constant diffThreshold value', 'diff threshold', 1, 255, nothing)    
    cv2.createTrackbar('variable diffThreshold value', 'diff threshold', 1, 100, nothing)   
    
    cv2.namedWindow('adaptive threshold', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('adaptiveThreshMaxValue', 'adaptive threshold', 0, 255, nothing)
    cv2.createTrackbar('adaptiveThreshBlockSize', 'adaptive threshold', 2, 20, nothing)
    cv2.createTrackbar('adaptiveThreshC', 'adaptive threshold', 0, 255, nothing)
    
    cv2.namedWindow('closing', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('closing K size', 'closing', 1, 100, nothing)
    
    cv2.namedWindow('allContours', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('smallContourSizeVar', 'allContours', 1,100, nothing)
    cv2.createTrackbar('smallContourSizeConst', 'allContours', 1,5000, nothing)
    cv2.createTrackbar('largeContourSizeVar', 'allContours', 1,100, nothing)
    cv2.createTrackbar('largeContourSizeConst', 'allContours', 5000,10000, nothing)    
    
    cv2.namedWindow('extent', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('extentThreshold', 'extent', 1,255, nothing)
    
    cv2.namedWindow('aspect', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('aspectThreshold', 'aspect', 1,255, nothing)    
    
    #cv2.namedWindow('largeContours', cv2.WINDOW_AUTOSIZE)
    #cv2.createTrackbar('largeContourThreshold', 'largeContours', 1,255, nothing)   
    
    cv2.namedWindow('largeContourThreshold',cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('largeContourThresholdValue', 'largeContourThreshold', 1,255, nothing)   
    
    
    count = 0
    
    while True:
        count += 1
        contourOnOriginal = image.copy()
        smallContours = image.copy()
        largeContours = image.copy()     
        problemChildren = image.copy()
        goodContours = image.copy()
        #largeContourMask = np.ones((height, width), dtype='uint8')*255
        largeContourMask = np.zeros((height, width), dtype='uint8')
        largeCountoursAlone = image.copy()
        smallContourMask = np.zeros((height, width), dtype='uint8')
        smallCountoursAlone = image.copy()     
        goodContourMask = np.zeros((height, width), dtype='uint8')
        goodContoursAlone = image.copy()
        extentContours = image.copy()
        aspectContours = image.copy()
        
        thresholdValue = cv2.getTrackbarPos('Threshold value', 'Threshold')
        
        erosionKernelSize = cv2.getTrackbarPos('erosion K size', 'erosion')
        erosionIterations = cv2.getTrackbarPos('erosion iterations', 'erosion')
        
        dilationKernelSize = cv2.getTrackbarPos('dilation K size', 'dilation')
        dilationIterations = cv2.getTrackbarPos('dilation iterations', 'dilation')        
        
        closingKernelSize = cv2.getTrackbarPos('closing K size', 'closing')
        
        numSplits = cv2.getTrackbarPos('number image splits', 'diff threshold')
        if numSplits < 1:
            numSplits = 1        
        diffThreshConstant = cv2.getTrackbarPos('constant diffThreshold value', 'diff threshold')
        diffThreshVar = cv2.getTrackbarPos('variable diffThreshold value', 'diff threshold')
        
        adaptiveThreshMaxValue = cv2.getTrackbarPos('adaptiveThreshMaxValue', 'adaptive threshold')
        adaptiveThreshBlockSize = (2*(cv2.getTrackbarPos('adaptiveThreshBlockSize', 'adaptive threshold')))-1
        if adaptiveThreshBlockSize < 3:
            adaptiveThreshBlockSize = 3
        adaptiveThreshC = cv2.getTrackbarPos('adaptiveThreshC', 'adaptive threshold')
        
        smallContourSizeVar = cv2.getTrackbarPos('smallContourSizeVar', 'allContours')
        smallContourSizeConst = cv2.getTrackbarPos('smallContourSizeConst', 'allContours')
        largeContourSizeVar = cv2.getTrackbarPos('largeContourSizeVar', 'allContours')
        largeContourSizeConst = cv2.getTrackbarPos('largeContourSizeConst', 'allContours')
               
        extentThreshold = float(cv2.getTrackbarPos('extentThreshold', 'extent'))/10
        aspectThreshold = float(cv2.getTrackbarPos('aspectThreshold', 'aspect'))/10   

        #largeContourThreshold = cv2.getTrackbarPos('largeContourThreshold', 'largeContours')   
        largeContourThresholdValue = cv2.getTrackbarPos('largeContourThresholdValue', 'largeContourThreshold')
        
        # hard coding values
        numSplits = 36
        diffThreshConstant = 177
        diffThreshVar = 1
        erosionKernelSize = 0
        erosionIterations = 3
        dilationKernelSize = 0
        dilationIterations = 3
        
        smallContourSizeVar = 1
        smallContourSizeConst = 339
        '''
        largeContourSizeVar = 15
        largeContourSizeConst = 5000
        '''
        
        grayGaus = cv2.GaussianBlur(gray, (3, 3), 0) # kernel has to be odd
        
        # basic threshold
        ret,thresh1 = cv2.threshold(grayGaus,thresholdValue,255,cv2.THRESH_BINARY_INV)
        
        # outsu threshold
        outsuThresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # adaptive thresholding. 
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C/cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        # THRESH_BINARY/THRESH_BINARY_INV
        adaptiveThresh = cv2.adaptiveThreshold(grayGaus, adaptiveThreshMaxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptiveThreshBlockSize, adaptiveThreshC)
        
        # differential thresholding
        deltaHeight = int(height/numSplits)
        for i in range(0, numSplits+1):
            minHeight = int(i*deltaHeight)
            maxHeight = int((i+1)*deltaHeight) 
            croppedImg = cropImage(grayGaus, minHeight, maxHeight)
            diffThreshValue = diffThreshConstant + diffThreshVar*(numSplits-i)
            ret,cropThreshImage = cv2.threshold(croppedImg,diffThreshValue,255,cv2.THRESH_BINARY)     
            reconstructedImage = reconstructImage(reconstructedImage, cropThreshImage, minHeight, maxHeight)
          
        diffThreshImage = reconstructedImage.copy()
        
        '''
        # watershed thresholding
        #diffThreshImage_inv = cv2.bitwise_not(diffThreshImage)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(diffThreshImage, cv2.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        
        cv2.imshow('opening', opening)
        cv2.imshow('sure_bg', sure_bg)
        cv2.imshow('sure_fg', sure_fg)
        cv2.imshow('dist_transform', dist_transform)
        cv2.imshow('unknown',unknown) 
        '''
        
        # morphology
        erosionKernel = np.ones((erosionKernelSize,erosionKernelSize),np.uint8)
        dilationKernel = np.ones((dilationKernelSize,dilationKernelSize),np.uint8)
        closingKernel = np.ones((closingKernelSize,closingKernelSize),np.uint8)
        
        dilation = cv2.dilate(diffThreshImage, dilationKernel, iterations=dilationIterations) # expands white object on black background
        erosion = cv2.erode(dilation, erosionKernel, iterations=erosionIterations) # reduces white object on black background
        #opening = cv2.morphologyEx(diffThreshImage, cv2.MORPH_OPEN, openingKernel) # expands any gaps in white objects
        closing = cv2.morphologyEx(diffThreshImage, cv2.MORPH_CLOSE, closingKernel) # reduces any gaps in white objects
        
        morphImage = erosion.copy()
        
        # contour detection
        contours, hierarchy = cv2.findContours(morphImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
        cv2.drawContours(contourOnOriginal, contours, -1, (255,0,0), 1)
        
        '''
        canny_output = cv2.Canny(morphImage, 100, 200)
        cv2.imshow('canny', canny_output)
        
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
        
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        
        for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.drawContours(drawing, contours_poly, i, color)
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
              (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
            
        cv2.imshow('Contours 2.0', drawing)        
        '''
        
        image_num = 0
        for j in range(len(contours)): 
            cnt = contours[j]        
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt,True)
                Rx,Ry,w,h = cv2.boundingRect(cnt)
                rect_area = w*h
                extent = float(area)/rect_area  
                aspect_ratio = float(w)/h   
                
                (CircleX,CircleY),radius = cv2.minEnclosingCircle(cnt)
                center = (int(CircleX),int(CircleY))
                radius_int = int(radius)
                circle_area_extent = float(area)/(math.pi*radius**2)
                
                text = "a %.2f\ny: %.2f\n"%(area,cy)
                #text = "extent: %.2f\n aspect: %.2f\narea: %.2f\nperimeter: %.2f\n" % (extent, aspect_ratio, area, perimeter)
                                            
                #cv2.putText(contourOnOriginal,text,(cx,cy), font, 0.2, (0,200,0), 1, cv2.LINE_AA)
                text_size = 0.3
                dy = int(text_size*25)
                for k, line in enumerate(text.split('\n')):
                    y = cy + k*dy
                    cv2.putText(contourOnOriginal, line, (cx, y), cv2.FONT_HERSHEY_SIMPLEX, text_size,(0,200,0), 1, cv2.LINE_AA)
                    
                if(aspect_ratio > aspectThreshold):
                    cv2.drawContours(aspectContours, contours, j, (255,0,0), 1)
                    
                if (extent > extentThreshold):
                    cv2.drawContours(extentContours, contours, j, (255,0,0), 1)
                
                # blobs too large for their area
                #if (area >= (-(height-cy)*0.5*largeContourSizeVar + largeContourSizeConst)):
                if ((area >= (11573*math.exp(-0.0115*(height-cy)))) and (circle_area_extent < 0.42)):
                    image_num += 1
                    #image_name = 'largeContour%d'%image_num
                    singleLargeContourMask = np.zeros((height, width), dtype='uint8') 
                    cv2.drawContours(singleLargeContourMask, contours, j, (255,0,0), -1)
                    singleLargeCountour = cv2.bitwise_and(image,image, mask=singleLargeContourMask)
                    cv2.imshow('singleLargeContour%d'%image_num,singleLargeCountour)
                    graySingleLargeCountour = cv2.cvtColor(singleLargeCountour, cv2.COLOR_BGR2GRAY) 
                    
                    smallEnough = 0
                    singleLargeContourThresholdValue = 1
                    while(not(smallEnough)):
                        ret,singleContourThresholded = cv2.threshold(graySingleLargeCountour,singleLargeContourThresholdValue,255,cv2.THRESH_BINARY_INV)
                        largeContoursAfterThreshold, hierarchy = cv2.findContours(singleContourThresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        #cv2.drawContours(singleContourThresholded, largeContoursAfterThreshold, -1, 0, 5)
                        cv2.imshow('singleContourThresholded',singleContourThresholded)
                        cv2.waitKey(0)
                        print 'largeContoursAfterThreshold: %d'%len(largeContoursAfterThreshold)
                        if(len(largeContoursAfterThreshold) == 1):
                            smallEnough = 1
                            print("small engouh")
                            break
                            
                        for largeCntCount in range(1,len(largeContoursAfterThreshold)): 
                            largeCnt = largeContoursAfterThreshold[largeCntCount]
                            M = cv2.moments(largeCnt)
                            if M["m00"] != 0:
                                area = cv2.contourArea(largeCnt)
                                cx = int(M['m10']/M['m00'])
                                cy = int(M['m01']/M['m00'])   
                                (CircleX,CircleY),radius = cv2.minEnclosingCircle(cnt)
                                center = (int(CircleX),int(CircleY))
                                radius_int = int(radius)
                                circle_area_extent = float(area)/(math.pi*radius**2)  
                                if((area >= (11573*math.exp(-0.0115*(height-cy)))) and (circle_area_extent < 0.42)):
                                    if (singleLargeContourThresholdValue < 150):
                                        singleLargeContourThresholdValue = singleLargeContourThresholdValue + 1 
                                        print 'singleLargeContourThresholdValue: %d'%singleLargeContourThresholdValue
                                else:
                                    print 'did the removing thing'
                                    # need to remove it from image
                                    removingLargeContourMask = np.ones((height, width), dtype='uint8')*255
                                    cv2.imshow('removingLargeContourMask before',removingLargeContourMask) # all white
                                    print 'largeCntCount: %d'%largeCntCount
                                    cv2.drawContours(removingLargeContourMask, largeContoursAfterThreshold, largeCntCount, 0, -1)
                                    cv2.imshow('removingLargeContourMask',removingLargeContourMask) # its blue??
                                    graySingleLargeCountour = cv2.bitwise_and(graySingleLargeCountour,graySingleLargeCountour,mask=removingLargeContourMask)
                                    
                                    # adding to good contours
                                    cv2.drawContours(goodContourMask, largeContoursAfterThreshold, largeCntCount, 255, -1)                                
                            else:
                                print("no M00 in the large contour parts")   
                                
                    cv2.drawContours(largeContours, contours, j, (255,0,0), 1)
                    cv2.circle(largeContours,(cx,cy), 2, (0,0,255), 1)
                    # CV_FILLED fills the connected components found
                    cv2.drawContours(largeContourMask, contours, j, 255, -1)
                    cv2.circle(problemChildren,center,radius_int,(255,255,255),2)  
                    
                    text = "extent: %.2f\naspect: %.2f\narea: %.2f\nperimeter: %.2f\nmoment:%.2f\ncircle:%.2f" % (extent, aspect_ratio, area, perimeter,M["m00"],circle_area_extent)
                    for k, line in enumerate(text.split('\n')): 
                        y = cy + k*dy
                        cv2.putText(problemChildren, line, (cx, y), cv2.FONT_HERSHEY_SIMPLEX, text_size,(0,200,0), 1, cv2.LINE_AA)
                # blobs too small for their area
                elif (area <= (-(height-cy)*smallContourSizeVar + smallContourSizeConst)): 
                    cv2.drawContours(smallContours, contours, j, (255,0,0), 1)
                    cv2.circle(smallContours,(cx,cy), 2, (0,0,255), 1)
                    cv2.drawContours(smallContourMask, contours, j, 255, -1)
                else:
                    cv2.drawContours(goodContours, contours, i, (255,0,0), 1)
                    cv2.circle(goodContours,(cx,cy), 2, (0,0,255), 1)
                    cv2.drawContours(goodContourMask, contours, j, 255, -1)
                    
            else:
                cv2.drawContours(problemChildren, contours, j, (255,0,0), 1)
        
        #largeContourMask2 = cv2.threshold(largeContourMask,255,255,cv2.THRESH_BINARY)
        largeCountoursAlone = cv2.bitwise_and(image,image, mask=largeContourMask)
        smallCountoursAlone = cv2.bitwise_and(image,image, mask=smallContourMask)
        goodCountoursAlone = cv2.bitwise_and(image,image, mask=goodContourMask)
        
        grayGoodCountoursAlone = cv2.cvtColor(goodCountoursAlone, cv2.COLOR_BGR2GRAY)
        finalPeopleContours, hierarchy = cv2.findContours(grayGoodCountoursAlone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
        cv2.drawContours(goodCountoursAlone, finalPeopleContours, -1, (255,0,0), 1)    
        numStudents = len(finalPeopleContours)
        
        grayLargeCountoursAlone = cv2.cvtColor(largeCountoursAlone, cv2.COLOR_BGR2GRAY)                
        HSVimageLargeCountoursAlone = cv2.cvtColor(largeCountoursAlone,cv2.COLOR_BGR2HSV)
        grayHSVLargeCountoursAlone = cv2.cvtColor(HSVimage, cv2.COLOR_BGR2GRAY)        
        ret,thresh2 = cv2.threshold(grayLargeCountoursAlone,largeContourThresholdValue,255,cv2.THRESH_BINARY_INV)
        
        
        # find large areas -> create bounding rectangle/circle/actual contour -> then threshold the shit outta them -> readd to image.         
        '''
        morphImage = cv2.bitwise_not(morphImage)     
        
        thresh1 = cv2.bitwise_not(thresh1)
        # blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = False
        #params.minArea = 30 # The dot in 20pt font has area of about 30
        params.filterByCircularity = False
        #params.minCircularity = 0.7
        params.filterByConvexity = False
        #params.minConvexity = 0.8
        params.filterByInertia = False
        #params.minInertiaRatio = 0.4
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thresh1)
        im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # laplace edge detection
        ddepth = cv2.CV_16S
        dst = cv2.Laplacian(morphImage, ddepth, 1)
        abs_dst = cv2.convertScaleAbs(dst)
        
        # laplace edge detection
        ddepth = cv2.CV_16S
        dst = cv2.Laplacian(image, ddepth, 1)
        abs_dst2 = cv2.convertScaleAbs(dst)
        '''
        
        # showing images
        cv2.imshow('Original', image)
        #cv2.imshow('HSV Original', HSVimage)
        #cv2.imshow('Gray', gray)
        #cv2.imshow('Gray Gaus', grayGaus)
        #cv2.imshow('Threshold', thresh1)
        #cv2.imshow('outsuThresh',outsuThresh)
        #cv2.imshow('diff threshold', diffThreshImage)
        #cv2.imshow('adaptive threshold', adaptiveThresh)       
        #cv2.imshow('erosion', erosion)
        #cv2.imshow('dilation', dilation)
        #cv2.imshow('closing', closing)
        #cv2.imshow('morphed image', morphImage)
        #cv2.imshow('Blobs Detected', im_with_keypoints)     
        #cv2.imshow('Laplacian', abs_dst)
        #cv2.imshow('Laplacian on origional', abs_dst2)
        cv2.imshow('contourOnOriginal', contourOnOriginal)
        allContours = np.concatenate((largeContours, smallContours, goodContours), axis=1)
        #cv2.imshow('allContours', allContours)
        cv2.imshow('problemChildren', problemChildren)
        #cv2.imshow('largeContourMask',largeContourMask)        
        cv2.imshow('largeCountoursAlone',largeCountoursAlone)
        #cv2.imshow('smallCountoursAlone', smallCountoursAlone)
        cv2.imshow('goodCountoursAlone',goodCountoursAlone)
        #cv2.imshow('extent', extentContours)
        #cv2.imshow('aspect', aspectContours)
        cv2.imshow('gray_raw',gray_raw)
        cv2.imshow('gray',gray)
        cv2.imshow('grayLargeCountoursAlone',grayLargeCountoursAlone)
        cv2.imshow('HSVimageLargeCountoursAlone',HSVimageLargeCountoursAlone)
        cv2.imshow('grayHSVLargeCountoursAlone',grayHSVLargeCountoursAlone)  
        cv2.imshow('largeContourThreshold',thresh2)
        
        if count % 10000:
            '''
            #print 'Threshold: ', thresholdValue
            print 'numSplits: ', numSplits
            print 'diffThreshConstant: ', diffThreshConstant
            print 'diffThreshVar: ', diffThreshVar            
            print 'erosionKernelSize: ', erosionKernelSize
            print 'erosion iterations: ', erosionIterations
            print 'dilation K size: ', dilationKernelSize
            print 'dilation iterations: ', dilationIterations
            print 'number of conotours: ', len(contours)
            '''
            '''
            print('contours:')
            print(contours)    
            print('single contour:')
            print(cnt)
            print('single area: ')
            print(area)
            
            print('largeContourSizeVar: ')
            print(largeContourSizeVar)  
            print('largeContourSizeConst: ')
            print(largeContourSizeConst) 
            print('smallContourSizeVar: ')
            print(smallContourSizeVar)  
            print('smallContourSizeConst: ')
            print(smallContourSizeConst)             
            print('length of contours: ')
            print(len(contours))
            '''
            print 'num of students: %d'%numStudents
            
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
if __name__ == "__main__":
    blobDetect()
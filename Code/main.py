import cv2
import numpy as np
from matplotlib import pyplot as plt
import random as rng
import math
import time

def nothing(x):
    pass

def cropImage(image, minHeight, maxHeight): 
    croppedImg = image[minHeight:maxHeight]
    return croppedImg

def reconstructImage(reconstructedImg, croppedThresh, minHeight, maxHeight):
    reconstructedImg[minHeight:maxHeight] = croppedThresh
    return reconstructedImg

def threshold():
    start = time.time()
    
    cv2.namedWindow('Threshold', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Threshold value', 'Threshold', 0, 255, nothing)
    
    cv2.namedWindow('HSV Threshold', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('HSV Threshold value', 'HSV Threshold', 0, 255, nothing)
    
    image = cv2.imread('../testImages/LeptonImages/A2_Monday13May_fullLecutre.png')# A5_half_full_thur_9_may
    height, width = image.shape[:2]
    
    testImage = image.copy()
    problemChildren = image.copy()
    allContoursOnMask = np.zeros((height, width), dtype='uint8') 
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reconstructedImage = gray.copy()
    
    HSVimage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    HSVgray = cv2.cvtColor(HSVimage, cv2.COLOR_BGR2GRAY)

    HSVgraySection = HSVimage[:,:,2]
    test = HSVimage[:,:,1]
    testThreshold = cv2.threshold(test,0,255,cv2.THRESH_BINARY)[1]
    cv2.imshow('test',test)
    cv2.imshow('testThreshold',testThreshold)
    
    '''
    HSVgrayThresholded = cv2.threshold(HSVgray,76,255,cv2.THRESH_BINARY)[1]
    cv2.imshow('HSVgrayThresholded before edit',HSVgrayThresholded)
    '''
    '''
    kernelSize = 1
    dilationIterations = 1 
    kernel = np.ones((kernelSize,kernelSize),np.uint8)    
    dilation = cv2.dilate(HSVgrayThresholded, kernel, iterations=1) # expands white object on black background
    cv2.imshow('dilation',dilation)
    erosion = cv2.erode(dilation, kernel, iterations=1) # reduces white object on black background
    cv2.imshow('erosion',erosion)
    '''
    #goodParts = cv2.bitwise_and(HSVgraySection,HSVgrayThresholded)
    
    goodParts = cv2.bitwise_and(HSVgraySection,testThreshold)
    cv2.imshow('HSVgraySection',HSVgraySection)
    #cv2.imshow('HSVgraySectionThresholded',HSVgrayThresholded)
    cv2.imshow('goodParts',goodParts)
    #cv2.waitKey(0)
    smallContourMask = np.zeros((height, width), dtype='uint8')
    goodContourMask = np.zeros((height, width), dtype='uint8')
    
    while True:
        
        thresholdValue = cv2.getTrackbarPos('Threshold value', 'Threshold') # makes some weird spots that can make it difficult to threshold (background and centre of faces are the same)
        HSVthresholdValue = cv2.getTrackbarPos('HSV Threshold value', 'HSV Threshold')
        
        #cv2.imshow('grayGaus',grayGaus)
        #cv2.imshow('HSVgraySection',HSVgraySection)
        #cv2.imshow('goodParts',goodParts)
        #cv2.imshow('outsuThresh',outsuThresh)
        #cv2.imshow('HSVoutsuThresh',HSVoutsuThresh)
              
        numSplits = 160
        diffThreshConstant = 0
        diffThreshVar = 1
        deltaHeight = float(height)/numSplits
        for i in range(0, numSplits+1):
            minHeight = int(i*deltaHeight)
            maxHeight = int((i+1)*deltaHeight) 
            croppedImg = cropImage(goodParts, minHeight, maxHeight)
            diffThreshValue = diffThreshConstant + diffThreshVar*(numSplits-i)
            ret,cropThreshImage = cv2.threshold(croppedImg,diffThreshValue,255,cv2.THRESH_BINARY)     
            reconstructedImage = reconstructImage(reconstructedImage, cropThreshImage, minHeight, maxHeight)
                  
        diffThreshImage = reconstructedImage.copy()        
        cv2.imshow('diffThreshImage',diffThreshImage)
        #closing2 = cv2.morphologyEx(diffThreshImage, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow('closing2',closing2)
        #cv2.waitKey(0)
        
        contours, hierarchy = cv2.findContours(diffThreshImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_num = 0
        for j in range(len(contours)): 
            image_num += 1
            subThresholdValue = 0
            singleLargeContourMask = np.zeros((height, width), dtype='uint8') 
            cv2.drawContours(singleLargeContourMask, contours, j, 255, -1)
            cv2.drawContours(allContoursOnMask, contours, j, 255, -1)
            
            singleLargeContour = cv2.bitwise_and(gray,gray, mask=singleLargeContourMask) 
            cv2.imshow('allContoursOnMask',allContoursOnMask)          
            
            contoursUnacceptable = 1
            while(contoursUnacceptable):
                singleContourThresholded = cv2.threshold(singleLargeContour,subThresholdValue,255,cv2.THRESH_BINARY)[1]
                cv2.imshow('singleContourThresholded',singleContourThresholded)
                # need to update singleLargeContour
                subContours, hierarchy = cv2.findContours(singleContourThresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                numSubContours = len(subContours)
                #cv2.imshow('singleLargeContour %d'%j,singleLargeContour)
                #cv2.imshow('singleContourThresholded %d'%j,singleContourThresholded)
                #cv2.waitKey(0)
                
                if (numSubContours == 0): 
                    print('breaking')
                    break
                else:
                    if (subThresholdValue < 255):
                        subThresholdValue += 1
                        print('subThresholdValue: %d'%subThresholdValue)
                        print('numSubContours: %d'%numSubContours)
                
                for k in range(numSubContours):
                    print('k: %d'%k)
                    subCnt = subContours[k]
                    M = cv2.moments(subCnt)
                    if M["m00"] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        area = cv2.contourArea(subCnt)
                        
                        Rx,Ry,w,h = cv2.boundingRect(subCnt)
                        rect_area = w*h
                        extent = float(area)/rect_area  
                        aspect_ratio = float(w)/h                           
                        
                        (CircleX,CircleY),radius = cv2.minEnclosingCircle(subCnt)
                        center = (int(CircleX),int(CircleY))
                        radius_int = int(radius)
                        circle_area_extent = float(area)/(math.pi*radius**2)
                        
                        hull = cv2.convexHull(subCnt)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area)/hull_area                        
                        
                        contourMask = np.zeros((height, width), dtype='uint8')
                        cv2.drawContours(contourMask, subContours, k, 255, -1)
                        removingContourMask = cv2.bitwise_not(contourMask)
                        
                        smallContourSizeVar = 0.92
                        smallContourSizeConst = 322                          
                        minArea = -0.8784*(height-cy)+306.73#-(height-cy)*smallContourSizeVar + smallContourSizeConst
                        
                        maxArea = 0.0559*(height-cy)**2-41.531*(height-cy)+7726.1#-13.74*(height-cy)+5516 #11573*math.exp(-0.0115*(height-cy))
                        minCircleExtent = 0.31  #0.42
                        
                        # blobs too small  
                        if (area <= minArea): 
                        #if ((area <= (-0.1265*(height-cy)+56.15)) or (area <= (-(height-cy)*smallContourSizeVar + smallContourSizeConst))):
                            # adding bad contour to small contour
                            cv2.drawContours(smallContourMask, subContours, k, 255, -1)
                            # remove from contour mask
                            singleLargeContour = cv2.bitwise_and(singleLargeContour,singleLargeContour,mask=removingContourMask)
                            print('\nwas too small\n')
                        # blobs not too large
                        elif ((area <= maxArea) and (circle_area_extent > minCircleExtent)):
                            # adding contour to good contour
                            cv2.drawContours(goodContourMask, subContours, k, 255, -1)
                            # remove from thresholding contour mask  
                            singleLargeContour = cv2.bitwise_and(singleLargeContour,singleLargeContour,mask=removingContourMask)
                            print('\nwas just right\n')
                            text = 'a:%.2f\ne:%.2f\ncy:%.2f'%(area,extent,(height-cy))
                            text_size = 0.2
                            dy = int(text_size*25)
                            for k, line in enumerate(text.split('\n')): 
                                y = cy + k*dy
                                cv2.putText(testImage, line, (cx, y), cv2.FONT_HERSHEY_SIMPLEX, text_size,(0,200,0), 1, cv2.LINE_AA)
                        elif (area >= maxArea):
                            print('area too large')
                        elif (circle_area_extent < minCircleExtent):
                            print('extent too large')
                        
                        contourBeingRemoved = cv2.bitwise_and(image,image, mask=contourMask)
                        
                        meanColour = cv2.mean(image, contourMask)
                        print('cy: %.2f'%cy)
                        print('area: %d'%area)
                        print('min area: %d'%minArea)
                        print('max area: %d'%maxArea)
                        print('min circle extent: %.2f'%minCircleExtent)
                        print('Circle extent: %.2f'%circle_area_extent)
                        print('Rectange extent: %.2f'%extent)
                        print('aspect_ratio: %.2f'%aspect_ratio)
                        print('solidity: %.2f'%solidity)
                        print('mean colour:',meanColour)
                        
                        print('%d;%.2f;%.2f;%.2f;%.2f;%.2f;%d;%d;%d'%(cy,area,circle_area_extent,extent,aspect_ratio,solidity,meanColour[0],meanColour[1],meanColour[2]))
                        #cv2.imshow('removingContourMask',removingContourMask)
                        #cv2.imshow('actual thing being removed', contourBeingRemoved)
                        #cv2.waitKey(0)  
                        
                    else:
                        cv2.drawContours(problemChildren, contours, k, (255,0,0), 1)
                        print('was a problem child')
        
        print('gone through all contours')
        smallCountoursAlone = cv2.bitwise_and(image,image, mask=smallContourMask)
        goodCountoursAlone = cv2.bitwise_and(image,image, mask=goodContourMask)

        grayGoodCountoursAlone = cv2.cvtColor(goodCountoursAlone, cv2.COLOR_BGR2GRAY)
        finalPeopleContours, hierarchy = cv2.findContours(grayGoodCountoursAlone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
        cv2.drawContours(goodCountoursAlone, finalPeopleContours, -1, (255,0,0), 1)  
        cv2.drawContours(image, finalPeopleContours, -1, (255,0,0), 1)
        numStudents = len(finalPeopleContours)        
        print('Student count: %d', numStudents)
        
        cv2.imshow('image',image)
        cv2.imshow('goodCountoursAlone',goodCountoursAlone)
        cv2.imshow('testImage',testImage)
        end = time.time()
        print('execution time: %f'%(end-start))
        cv2.waitKey(0)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    
if __name__ == "__main__":
    threshold()
import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

def threshold():
    img = cv2.imread('../testImages/FLIRImages/thermal3.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Image Split', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Threshold Constant', 'Image Split', 0, 255, nothing)
    cv2.createTrackbar('Threshold Variable', 'Image Split', 0, 255, nothing)

    while True:
        thresholdValue = cv2.getTrackbarPos('Threshold value', 'Threshold Transform')
        
        ret,thresh1 = cv2.threshold(gray,thresholdValue,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
        ret,thresh3 = cv2.threshold(gray,thresholdValue,255,cv2.THRESH_MASK)
        ret,thresh4 = cv2.threshold(gray,0,255,cv2.THRESH_TRIANGLE)
        ret,thresh5 = cv2.threshold(gray,thresholdValue,255,cv2.THRESH_TOZERO_INV)        
        '''
        cv2.imshow('Binary', thresh1)
        cv2.imshow('Binary Inverse', thresh2)
        cv2.imshow('Trunc', thresh3)
        cv2.imshow('To Zero', thresh4)
        cv2.imshow('To Zero Inverse', thresh5)        
        '''
        # img.shape[0] -> height   
        # img.shape[1] -> width
        width = 243 
        height = int(width*(img.shape[0]/img.shape[1]))
        dim = (width, height)

        thresh1 = cv2.resize(thresh1, dim)
        thresh2 = cv2.resize(thresh2, dim)
        thresh3 = cv2.resize(thresh3, dim)
        thresh4 = cv2.resize(thresh4, dim)
        thresh5 = cv2.resize(thresh5, dim)
        
        combined = np.concatenate((thresh1, thresh2, thresh3, thresh4, thresh5), axis=1)
        cv2.imshow('Threshold Transform', combined)     

        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    threshold()
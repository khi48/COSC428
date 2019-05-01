import cv2
import numpy as np

def nothing(x):
    pass

def smoothing():
    img = cv2.imread('../testImages/FLIRImages/thermal3.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Smoothing Transforms', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Gaussian Kernel Value', 'Smoothing Transforms', 1, 100, nothing)
    cv2.createTrackbar('Smoothing Iterations', 'Smoothing Transforms', 0, 100, nothing)
    
    while True:
        gausKernelSize = cv2.getTrackbarPos('Gaussian Kernel Value', 'Smoothing Transforms')//2*2+1
        gausIterations = cv2.getTrackbarPos('Smoothing Iterations', 'Smoothing Transforms')
                
        # smoothing features        
        gray_blur = cv2.GaussianBlur(gray, (gausKernelSize, gausKernelSize), gausIterations)
        
        adaptiveThresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
        
        '''
        cv2.imshow('Threshold', binaryThresh)
        cv2.imshow('erosion', erosion)
        cv2.imshow('dilation', dilation)
        cv2.imshow('opening', opening)
        cv2.imshow('closing', closing)
        '''
        
        # Displaying Image
        combined = np.concatenate((gray_blur, adaptiveThresh), axis=1)
        cv2.imshow('Smoothing Transforms', combined)         
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    smoothing()        
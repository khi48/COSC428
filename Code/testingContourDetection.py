import cv2
import numpy as np
from matplotlib import pyplot as plt
import random as rng
import math
height = 640
width = 480
img = np.ones((height, width), dtype='uint8')*255
img2 = np.ones((height, width), dtype='uint8')*255
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
cv2.drawContours(img2, contours, -1, 0, 1)
print(len(contours))
cv2.imshow('img',img)
cv2.imshow('img2',img2)
cv2.waitKey(0)
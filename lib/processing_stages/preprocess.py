import cv2
import numpy as np

def canny(image,
          gaussian={'ksize': (15, 15), 'sigmaX': 0},
          canny={'threshold1': 25, 'threshold2': 40, 'L2gradient': True},
          dilate={'kernel': np.ones((3, 3))},
          erode={'kernel': np.ones((5, 5))}):
    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.GaussianBlur(processed, **gaussian)
    processed = cv2.Canny(processed, **canny)
    processed = cv2.dilate(processed, **dilate)
    #processed = cv2.erode(processed, **erode)
    return processed

def threshold(image,
              gaussian={'ksize': (15, 15), 'sigmaX': 0},
              threshold={'thresh': 160, 'maxval': 255, 'type': cv2.THRESH_BINARY}):
    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.GaussianBlur(processed, **gaussian)
    processed = cv2.threshold(processed, **threshold)[1]
    return processed

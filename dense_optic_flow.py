# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:01:03 2023

@author: Cecilia H. Hansen
"""


#Code from https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/ which has been modified to fit the purpose of this project
import cv2
import numpy as np
from skimage.io import imread
  
def opticflow(videoframes):
    maglist=[]
    magdict={}
    numberlist=range(videoframes.shape[0])
    # The video feed is read in as
    # a VideoCapture object
    #cap = cv2.VideoCapture("InputVideosSplit/train/PolypsBetween6and10mm/13403-P1.mpg")
      
    # ret = a boolean return value from
    # getting the frame, first_frame = the
    # first frame in the entire video sequence
    #_, first_frame = cap.read()
    first_frame = np.float32(videoframes[0]) #we have floats in the image and convert it


      
    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally 
    # expensive
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    #cv.imshow("input", prev_gray)
    #cv.waitKey(0)
    # Creates an image filled with zero
    # intensities with the same dimensions 
    # as the frame
    mask = np.zeros_like(first_frame)
      
    # Sets image saturation to maximum
    mask[..., 1] = 255
      
    while(True):
        
          
        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        #ret, frame = cap.read()
        for number, frame in zip(numberlist, videoframes):
            
            frame=np.float32(frame)
            
          
        # Opens a new window and displays the input
        # frame
        #cv2.imshow("input", frame)
          
        # Converts each frame to grayscale - we previously 
        # only converted the first frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
              
            # Calculates dense optical flow by Farneback method
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                               None,
                                               0.5, 3, 15, 3, 5, 1.2, 0)
              
            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            maglist.append([np.max(magnitude), np.median(magnitude), np.mean(magnitude)])
            magdict[number]=maglist
            maglist=[]
        return magdict
                  
# =============================================================================
#             # Sets image hue according to the optical flow 
#             # direction
#             mask[..., 0] = angle * 180 / np.pi / 2
#               
#             # Sets image value according to the optical flow
#             # magnitude (normalized)
#             mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
#               
#             # Converts HSV to RGB (BGR) color representation
#             rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
#               
#             # Opens a new window and displays the output frame
#             cv2.imshow("dense optical flow", rgb)
#               
#             # Updates previous frame
#             prev_gray = gray
#               
#             # Frames are read by intervals of 1 millisecond. The
#             # programs breaks out of the while loop when the
#             # user presses the 'q' key
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#       
# # The following frees up resources and
# # closes all windows
# cap.release()
# cv2.destroyAllWindows()
# =============================================================================

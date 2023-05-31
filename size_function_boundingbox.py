#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:28:17 2023

@author: louisethomsen
Modified to fit own code by Cecilia Holm Hansen 15.05.2023
"""





import cv2
import numpy as np
import os
import re
import pandas as pd
from skimage.io import imread

def size_function(img):
    img_rot_width = []
    img_rot_height = []
    highest_value = []
    number_of_contours = []
    #Loading the image
    img1 = (img * 255).astype(np.uint8) #converting the image to a 8bit image  where pixelvalues are between 0-255 instead of 0-1
    # convert the image to grayscale
    if img1.shape == (576,576,3):
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    ret,thresh = cv2.threshold(img1,127,255,0) # For every pixel, the same threshold value is applied. If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value. The function cv.threshold is used to apply the thresholding. The first argument is the source image, which should be a grayscale image. The second argument is the threshold value which is used to classify the pixel values. The third argument is the maximum value which is assigned to pixel values exceeding the threshold.
    
    #using the threshold value to find the contours in the image (where there is black pixels surrounding white pixel(s))
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html there are three arguments in cv.findContours() function, first one is source image, second is contour retrieval mode, third is contour approximation method. And it outputs the contours and hierarchy. Contours is a Python list of all the contours in the image. Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.

    #counting how many contours and appending them to a list
    Len_contours = len(contours)
    number_of_contours.append(Len_contours)
    
    #If there are contours in the list make another empty list
    if Len_contours != 0: 
        areaArray = []

        for i, c in enumerate(contours):
            area = cv2.contourArea(c) 
            areaArray.append(area) #append the area of the contour to new list
            areaLargest = np.argmax(areaArray) #find the largest area in new list (some images can have smaller white pixels)

        cnt = contours[areaLargest] #input the largest contour
        
        # compute straight bounding rectangle
        x,y,w,h = cv2.boundingRect(cnt) #make a rectangle and define center, weight and height of it
        img = cv2.drawContours(img,[cnt],0,(255,255,0),2) #draw the contour over the image
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #draw rectangle on image
        
        # compute rotated rectangle (minimum area)
        rect = cv2.minAreaRect(cnt) #(center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect) #makes the box points from the rotated rectangle
        box = np.int0(box)
        
        # draw minimum area rectangle (rotated rectangle)
        img = cv2.drawContours(img,[box],0,(0,255,255),2) #draw rotated rectangle

        img_rot_width.append(rect[1][1]) #appends the width of the rotated rectangle to list (w)
        img_rot_height.append(rect[1][0])
    
        #Here we check to see if either the height or width has the highest value - the highest value must be the actual width (diameter at the broadest point of the polyp)
        if rect[1][1] > rect[1][0]: 
            highest_value.append(rect[1][1])
        else: 
            highest_value.append(rect[1][0])
    #if there are no contours we just append 0 to the lists
    else:
        img_rot_width.append(0)
        img_rot_height.append(0)
        highest_value.append(0) #largest diameter
    return highest_value #the actual polyp diameter


test_image=r"C:\Users\Cecilia H. Holm\Documents\Speciale\Finished Annotations\10032\P1\PixelLabelData\Label_1.png"
im =imread(test_image)

im.shape
size_function(im)

#WITH MORE PARAMETERS


def polyp_features(img):
    
    #img = cv2.imread(np.squeeze(img))
    
    #scaling the image
    img = (img * 255).astype(np.uint8) #making 8 bit image/array  
    #convert the image to grayscale
    #if(len(img1.shape)>2):
    #    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    

    ret,thresh = cv2.threshold(img,127,255,0) # Making a binary image (black/white pixels) based on a threshold value of the pixels. The first argument is the source image, which should be a grayscale image. The second argument is the threshold value which is used to classify the pixel values. The third argument is the maximum value which is assigned to pixel values exceeding the threshold. OpenCV provides different types of thresholding which is given by the fourth parameter of the function. https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find the contours around white pixels based on the threshold image (black/white image). Each contour is stored as a vector of points

    num_of_contours = len(contours)  #count number of contours found in image
    
    
    if num_of_contours != 0: 
        #Checks if there are multiple contours on the image, if so choose the largest
        areaArray = []
        for i, c in enumerate(contours):
            area = cv2.contourArea(c) #area of contour
            areaArray.append(area) #append to list
            areaLargest = np.argmax(areaArray) #find largest area in image
        cnt = contours[areaLargest] 
        
        
        ###Contour data 
        #Contour area
        contour_area = cv2.contourArea(cnt) #calculate the area of the largest white contour on image
        
        #Contour circumference
        contour_circumference = cv2.arcLength(cnt,True) 
        
    
        ###Bounding box 
        
        #Compute straight bounding rectangle
        x,y,w,h = cv2.boundingRect(cnt) #take the coordinates (and height and width) for the box to draw a square around it
        img = cv2.drawContours(img,[cnt],0,(255,255,0),2)
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
        #Compute rotated rectangle (minimum area)
        rect = cv2.minAreaRect(cnt) 
        box = cv2.boxPoints(rect) #box is the four corners of the rectangle
        box = np.int0(box)
        
        # Extract box size and angle from tuple
        box_center, box_size, box_angle = rect
        box_width, box_height = box_size
        
        #Draw rotating bounding box
        img = cv2.drawContours(img,[box],0,(0,255,255),2)
        
        
        
        #Define smallest diameter and largest diameter (the function doesn't know whats the width and what is the height). We define the largest diameter as the box width)
        if box_width > box_height:
            largest_diameter = box_width
            smallest_diameter = box_height
        else: 
            largest_diameter = box_height
            smallest_diameter = box_width
        
        #Box area
        box_area = box_width * box_height
        
        #Extent (object/box area)
        if box_area==0.0:
            box_area=1
            extent_fill = contour_area/box_area
        else:
            extent_fill = contour_area/box_area

      
    else: #if the image has no contour of a polyp
        largest_diameter = 0
        smallest_diameter = 0
        extent_fill = 0
        contour_area = 0
        contour_circumference = 0
        
    return contour_area, contour_circumference, largest_diameter, smallest_diameter, extent_fill
        
im=r"C:\Users\Cecilia H. Holm\Documents\Speciale\Finished Annotations\10032\P1\PixelLabelData\Label_1.png"
im =cv2.imread(im)
im.shape
polyp_features(im)
type(im)
im.dtype

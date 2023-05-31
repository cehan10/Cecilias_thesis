# -*- coding: utf-8 -*-
"""
Created on Sat May 20 12:28:24 2023

@author: Cecilia H. Hansen
"""
import pandas as pd
g = pd.read_pickle(r"C:\Users\Cecilia H. Holm\Documents\Speciale\final_df.pkl")


# importing 
import os
from pathlib import Path
import shutil
import pickle
import pandas as pd
import splitfolders #https://pypi.org/project/split-folders/

#%% ONLY HAS TO RUN ONCE! GOING INTO EACH PATIENT FOLDER AND EXTRACTING THE VIDEOFILES FROM THE RELEVANT POLYPS (5 IMAGES AND HAS A PATHOLGY SIZE)
  
#Making new empty directories based on the categories (pathology size categories)
# Directory
directory = "PolypsUnder6mm"
# Parent Directory path
parent_dir = "C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideos/"
# Path
path = os.path.join(parent_dir, directory)
# Create the directory
os.mkdir(path)

directory='PolypsBetween6and10mm'
path = os.path.join(parent_dir, directory)
# Create the directory
os.mkdir(path)

directory='PolypsBetween10and20mm'
path = os.path.join(parent_dir, directory)
# Create the directory
os.mkdir(path)

directory='PolypsOver20mm'
path = os.path.join(parent_dir, directory)
# Create the directory
os.mkdir(path)



root1 = 'C:/Users/Cecilia H. Holm/Documents/Speciale/Finished Annotations'
dir1='C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideos/PolypsUnder6mm'
dir2='C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideos/PolypsBetween6and10mm'
dir3='C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideos/PolypsBetween10and20mm'
dir4='C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideos/PolypsOver20mm'


video_list=[]

#GETTING THE VIDEOFILES INTO SEPERATE FOLDERS BASED ON CATEGORIES
           
def divide_dataset(root, dataframe, dir1, dir2, dir3, dir4, videolist):       
    #count number of patient directories
    for folders in os.listdir(root):
        #Looking through each patient directory
        for direc in os.listdir(root+'/'+folders): 
            if direc.startswith(('P1','P2','P3','P4','P5','P6','P7','P8','P9')): #Look through Polyp folders
                    for ro in os.listdir(root+'/'+folders+'/'+direc): #look through the polyp directory 
                        if ro.startswith(('Raw', 'raw')): #look in the folder for Raw images folder
                            for file in os.listdir(root+'/'+folders+'/'+direc+'/'+ro):
                                if (folders, direc) in dataframe.index: #look for the patient, polyp number in index in dataframe    
                                    if file.endswith('.mpg'):
                                        if (dataframe.loc[(folders, direc), 'Pathology polyp size']<6).any():
                                            old_name= root+'/'+folders+'/'+direc+'/'+ro+'/'+file
                                            new_name=dir1+'/'+folders+'-'+direc+'.mpg'
                                            videolist.append(folders+'-'+direc+'.mpg')
                                            #os.rename(old_name, new_name)
                                            shutil.copy(old_name, new_name)
                                        elif (dataframe.loc[(folders, direc), 'Pathology polyp size']>=6).any() & (dataframe.loc[(folders, direc), 'Pathology polyp size']<10).any():
                                            old_name= root+'/'+folders+'/'+direc+'/'+ro+'/'+file
                                            new_name=dir2+'/'+folders+'-'+direc+'.mpg'
                                            #os.rename(old_name, new_name)
                                            videolist.append(folders+'-'+direc+'.mpg')
                                            shutil.copy(old_name, new_name)
                                        elif (dataframe.loc[(folders, direc), 'Pathology polyp size']>=10).any() & (dataframe.loc[(folders, direc), 'Pathology polyp size']<20).any():
                                            old_name= root+'/'+folders+'/'+direc+'/'+ro+'/'+file
                                            new_name=dir3+'/'+folders+'-'+direc+'.mpg'
                                            #os.rename(old_name, new_name)
                                            videolist.append(folders+'-'+direc+'.mpg')
                                            shutil.copy(old_name, new_name)
                                        elif (dataframe.loc[(folders, direc), 'Pathology polyp size']>=20).any():
                                            old_name= root+'/'+folders+'/'+direc+'/'+ro+'/'+file
                                            new_name=dir4+'/'+folders+'-'+direc+'.mpg'
                                            #os.rename(old_name, new_name)
                                            videolist.append(folders+'-'+direc+'.mpg')
                                            shutil.copy(old_name, new_name)
                                            
                                        else: 
                                            continue
                                    elif not any(file.endswith('.mpg') for file in os.listdir(root+'/'+folders+'/'+direc+'/'+ro)):
                                            print(folders,direc)
                                    else:
                                        continue
                                else:
                                    continue
                        else:
                            continue
                    else:
                        continue

                    
divide_dataset(root1, g, dir1, dir2, dir3, dir4, video_list)
#One folder doesn't have a video (35499 PI)


# SPLITTING THE DATASET INTO TEST AND TRAIN FOLDERS


#DATAPREP
input_folder='C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideos/'

#Splitting the data into training and validation(test) set
splitfolders.ratio(input_folder, output="InputVideosSplit", seed=42, ratio=(0.8, 0.1, 0.1)) 
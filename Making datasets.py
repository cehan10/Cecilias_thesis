# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:50:32 2023

@author: Cecilia H. Hansen
"""
import os
import cv2
import pandas as pd
from size_function_boundingbox import size_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import seaborn as sb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from skimage.io import imread
import statsmodels.api as sm
import pylab as py


#%% Function 

per_patient_stats = {}
polypsdictionary={}

#dictionary={}
root1 = 'C:/Users/Cecilia H. Holm/Documents/Speciale/Finished Annotations'

           
def explore_annotations(root, dictionary, polypsperpatient):       
    #count number of patient directories
    for folders in os.listdir(root):
        dictionary[folders]={} #making a dictionary in the dictionary for each patient
        polypsperpatient[folders]={}
        #Making lists and dictionaries over the different elements to later put into the dictionary
        list_of_polyps=[] 
        #list_of_negativelabels=[]
        polypsdict={}
        #Looking through each patient directory
        for direc in os.listdir(root+'/'+folders): 
            if direc.startswith(('P1','P2','P3','P4','P5','P6','P7','P8','P9')): #Look through Polyp folders
                list_of_polyps.append(direc) #put the directory name into list
                polypsperpatient[folders]['polyps']=len(list_of_polyps)
                for ro in os.listdir(root+'/'+folders+'/'+direc): #look through the polyp directory 

                    if ro.startswith(('Pixel', 'pixel')): #if folder starts with pixel (where the labels are located)
                       
                    #making lists over polyps an no polyps
                        label_polyp=[]
                        label_nopolyp=[]
                        for file in os.listdir(root+'/'+folders+'/'+direc+'/'+ro): #look on label folder
                            if file.startswith('Label'): #look for label images
                                image=cv2.imread(os.path.join(root+'/'+folders+'/'+direc+'/'+ro+'/'+file)) #read the image
                                image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #make sure the image is black/white 8bit grayscale
                                if cv2.countNonZero(image) == 0: #look if there are non-zero pixel values in the image array
                                    label_nopolyp.append(file) #if not append to no-polyp list
                                else:
                                    label_polyp.append(file) #if there is, append to polyp list
                            else:
                                continue
                    elif ro.startswith(('Raw', 'raw')): #look in the folder for Raw images folder
                        images_in_each_folder=[] #make an empty list
                        for file in os.listdir(root+'/'+folders+'/'+direc+'/'+ro):#look in the folder
                            if file.endswith('.jpg'): #for jpg files (images of polyps)
                                images_in_each_folder.append(file) #append image names to list
                            else:
                                continue
                         #Make an empty dictionary for each polyp    
                        polypsdict[direc]={}
                         #The number of images in each raw image folder (how many images are there of each polyp)
                        polypsdict[direc]['Number of images in folder']=len(images_in_each_folder)
                         #The number of positive labels in each label folder and put it into the patient dictionary
                        polypsdict[direc]['Number of labels=polyp in folder']=len(label_polyp)
                         #The number of negative labels in each label folder and put it into the patient dictionary
                        polypsdict[direc]['Number of labels=no polyp in folder']=len(label_nopolyp)
                         #Add the polypsdictionary into the patient folder of the main dictionary
                        dictionary[folders]=polypsdict
                        
                        #Emptying the lists after use
                        label_polyp=[]
                        label_nopolyp=[]
                        images_in_each_folder=[]
                    else:
                        continue   
            else:                
                continue
            
    return dictionary, polypsperpatient
            

explore_annotations(root1, per_patient_stats, polypsdictionary)   

#Making the innerkeys(polyp number - fx P1) an index together with the outerkeys (patient)
statsdf={}
for outerkey, innerDict in per_patient_stats.items():
    for innerkey, values in innerDict.items():
        statsdf[(outerkey, innerkey)]=values
#Making the dictionary into a dataframe and transposing it so the index is on the left
statsdf=pd.DataFrame(statsdf).T
statsdf.index.names=['Patient', 'Polyp'] #Making names for the indexes

#Making the other dictionary into a dataframe
n_polyps=pd.DataFrame(polypsdictionary).T

#%%STATISTICS
#How many polyps do the patients have?
n_polyps.groupby('polyps').size()

n_polyps.describe() #598 patients - one folder is empty

statsdf.groupby('Number of images in folder').size()
statsdf.groupby('Number of labels=polyp in folder').size()
statsdf.groupby('Number of labels=no polyp in folder').size()

 

statsdf.to_csv('statsdf.csv',index=True)  
n_polyps.to_csv('n_polyps.csv',index=True)  


#%% IMPORT SIZE DOCUMENT TO COMPARE AND MERGE THE TWO DATAFRAMES

#Size = pd.read_excel(r"C:\Users\Cecilia H. Holm\Documents\Speciale\Size\Matched polyps 14-11-2022.xls")
Size = pd.read_excel(r"C:\Users\Cecilia H. Holm\Documents\Speciale\Size\Matched polyps 03-05-2023.xls")
Size['CCE polyp no'] ='P' + Size['CCE polyp no'].astype(str)
Size['CCE polyp no'] =Size['CCE polyp no'].astype(str)
Size['CCE polyp no'] =Size['CCE polyp no'].astype(object)
Size['SDK-ID'] =Size['SDK-ID'].astype(str)
Size['SDK-ID'] =Size['SDK-ID'].astype(object)
Size=Size.set_index(['SDK-ID', 'CCE polyp no'])
Size.index.names=['Patient', 'Polyp'] #Making names for the indexes
result = pd.merge(Size, statsdf, on=["Patient", "Polyp"])

result.groupby('Number of images in folder').size()
df = result[result['Number of images in folder']== 5] #we only want folders with 5 raw images
df = result[result['Number of labels=polyp in folder']== 5] #we only want folders with 5 labels = polyp
df.to_csv('five_image_folders.csv',index=True)


#%% GO INTO FOLDER AND COLLECT BOUNDING BOX WIDTH

sizedict={}
def bounding_box(root, dataframe, dictionary):  
    size_list=[]
    #count number of patient directories
    for folders in os.listdir(root):
        if folders in dataframe.index.levels[0]:
            dictionary[folders]={}
            polypsdict={}
            for direc in os.listdir(root+'/'+folders): 
                if (folders, direc) in dataframe.index:
                    size_list=[]
                #if direc.startswith(('P1','P2','P3','P4','P5','P6','P7','P8','P9')): #Look through Polyp folders
                    for ro in os.listdir(root+'/'+folders+'/'+direc): #look through the polyp directory 
                        if ro.startswith(('Pixel', 'pixel')): #if folder starts with pixel (where the labels are located)
                            for file in os.listdir(root+'/'+folders+'/'+direc+'/'+ro): #look on label folder
                                if file.startswith('Label'): #look for label images
                                    image=imread(os.path.join(root+'/'+folders+'/'+direc+'/'+ro+'/'+file)) #read the image
                                    #print(image.shape)
                                    pixel_size=size_function(image)
                                    size_list.append(pixel_size)
                                    
                                else:
                                    continue
                            polypsdict[direc]={}
                            #Put each of the numbers into the corresponding image number cell in the dictionary 
                            polypsdict[direc]['Image1']=size_list[0]
                            polypsdict[direc]['Image2']=size_list[1]
                            polypsdict[direc]['Image3']=size_list[2]
                            polypsdict[direc]['Image4']=size_list[3]
                            polypsdict[direc]['Image5']=size_list[4]
                            dictionary[folders]=polypsdict
                            
                            #Emptying the lists after use
                            size_list=[]
                                
                        else:
                            continue   
                else:                
                    continue
        else:
            continue
            
    return dictionary
    
bounding_box(root1, df, sizedict)

image_sizedict={}
for outerkey, innerDict in sizedict.items():
    for innerkey, values in innerDict.items():
        image_sizedict[(outerkey, innerkey)]=values
#Making the dictionary into a dataframe and transposing it so the index is on the left
image_sizedict=pd.DataFrame(image_sizedict).T
image_sizedict.index.names=['Patient', 'Polyp'] #Making names for the indexes
image_sizedict=image_sizedict.applymap(lambda x: x[0] if isinstance(x, list) else x) #because the cells contain lists, we remove the lists and keep their values (so we get numbers instead)
imagestats=pd.DataFrame(image_sizedict.T.describe())
#Difference between largest and smallest value
#diffdf=pd.DataFrame(image_sizedict.T.max()-image_sizedict.T.min())
#imagestats=pd.merge(imagestats.T, diffdf, on=["Patient", "Polyp"])
#imagestats.rename(columns = {0:'min_max_diff'}, inplace = True)
g1=df[['Pathology polyp size', 'Histology']]
#g1=pd.merge(diffdf,g1, on=["Patient", "Polyp"])
g=pd.merge(image_sizedict,g1, on=["Patient", "Polyp"])
#g=pd.merge(g1,imagestats, on=["Patient", "Polyp"])
#g.rename(columns = {0:'min_max_diff'}, inplace = True)

g=g.dropna() #dropping cells where the pathology size is=na. This is 38 rows/polyps. Leaving 138 polyps.
g.to_csv('final_df.csv',index=True)  

#These values tell us that if the Pathology size increase by 1mm??, the pixels will increase with the numbers given in the reg coefficient
nlo=g1.drop(['Histology'], axis=1)
nlo=nlo.corr() #we see that the strongest correlation is between image 3 and pathology size (0.708)
sb.heatmap(nlo, annot=True, cmap='mako')
plt.show()

#np.polyfit(g['Pathology polyp size'],g['Image1'], deg=1)




#%%ADDING Categories to the dataframe and saving it
def categorise(row):  
    if row['Pathology polyp size'] <6:
        return '0-5mm'
    elif row['Pathology polyp size'] >= 6 and row['Pathology polyp size'] < 10:
        return '6-9mm'
    elif row['Pathology polyp size'] >= 10  and row['Pathology polyp size'] < 20:
        return '10-19mm'
    elif row['Pathology polyp size'] >= 20:
        return 'Over 20 mm'

g['Category'] = g.apply(lambda row: categorise(row), axis=1)


g.to_pickle('final_df.pkl') 
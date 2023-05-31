# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:09:11 2023

@author: Cecilia H. Hansen
"""
import os

# import opencv
import cv2
import pandas as pd
from size_function_boundingbox import size_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import seaborn as sb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
import pylab as py
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import PredictionError, ResidualsPlot
import pandas as pd
from math import sqrt
from sklearn.metrics import confusion_matrix

g = pd.read_pickle(r"C:\Users\Cecilia H. Holm\Documents\Speciale\final_df.pkl")

#%% PLOTS OF EACH IMAGE AS LINEAR REGRESSION



def plotting(sizedataframe, number):
    gt=sizedataframe#[sizedataframe['Pathology polyp size']<=20]  
    regr = linear_model.LinearRegression() #to force intercept to 0 =>NOT EQUAL to plot Set fit_intercept=False
    #regr = linear_model.LinearRegression()
    X=gt['Image'+str(number)]
    X=X.values.reshape(-1, 1)
    y=gt['Pathology polyp size']
    regr.fit(X, y)
    sb.lmplot( data=gt, x='Image'+str(number), y='Pathology polyp size', fit_reg=True, ci=None, scatter_kws={"marker": "D", "s": 20})
    plt.title("Image" +str(number)+" size vs Pathology size")
    plt.ylabel("Pathology size")
    plt.xlabel("Image" +str(number)+" pixel size")
    plt.annotate("R^2 = {:.3f}".format(regr.score(X, y)), (50,65))
    plt.annotate("y = 0.074x-3.836", (50,60))
    plt.show()
    print(regr.intercept_,float(regr.coef_))
    return regr  


plotting(g, 3)


#%% calculating the polyp mm from the intercept and slope

#polyp_mm = intercept + slope × polyp_px

def calc_mm(linreg, sizedataframe, number):
    mmlist=[]
    for i in sizedataframe['Image'+str(number)]:
        sizemm = linreg.intercept_ + float(linreg.coef_ ) * i
        #sizemm=(i/ float(linreg.coef_ )) - linreg.intercept_
        mmlist.append(sizemm)
    sizedataframe['Calculated size mm image '+str(number)]=mmlist
    sizedataframe['diff image 3']=sizedataframe['Pathology polyp size']-sizedataframe['Calculated size mm image '+str(number)]
    return sizedataframe

calc_mm(plotting(g,3), g, 3) 



#%% Multivariate regression
#Independent variables = image sizes in pixels
#Dependent variable = pathology image size


#Multivariate linear model 
X=g[['Image1','Image2','Image3','Image4','Image5']].values
y1=g['Pathology polyp size'].values
regr1 = linear_model.LinearRegression() #to force intercept to 0 =>NOT EQUAL to plot Set fit_intercept=False
regr1.fit(X, y1 )
print(regr1.intercept_,regr1.coef_)




#%%
#### CALCULATE THE PREDICTED SIZE FROM THE REGRESSION #######
def calc_mmpoly(linreg, sizedataframe):
    mmlist=[]
    for row in sizedataframe.itertuples(index=False, name='Pandas'):
        sizemm = linreg.intercept_ + linreg.coef_[0]  * getattr(row, "Image1") + linreg.coef_[1]  * getattr(row, "Image2") + linreg.coef_[2]  * getattr(row, "Image3") + linreg.coef_[3]  * getattr(row, "Image4") + linreg.coef_[4]  * getattr(row, "Image5")
        #sizemm=(i/ float(linreg.coef_ )) - linreg.intercept_
        mmlist.append(sizemm)
    sizedataframe['Calculated size mm image']=mmlist
    sizedataframe['diff image multi']=sizedataframe['Pathology polyp size']-sizedataframe['Calculated size mm image']
    return sizedataframe

calc_mmpoly(regr1, g) 

#%%
####PLOTTING OF CALCULATED/PERDICTED SIZE OVER ACTUAL SIZE #######
def plotting2(sizedataframe): 
    X=sizedataframe['Calculated size mm image']
    X=X.values.reshape(-1, 1)
    y=sizedataframe['Pathology polyp size'].values
    regr2 = linear_model.LinearRegression()
    regr2.fit(X, y )
    print(regr2.intercept_,regr2.coef_)
    sb.regplot(x=X, y=y, ci=None)
    #sizedataframe.plot(kind='scatter', x='Calculated size mm image', y='Pathology polyp size',color='DarkBlue')
    #sb.lmplot( data=gt, x='Calculated size mm image', y='Pathology polyp size', fit_reg=True, scatter_kws={"marker": "D", "s": 20})
    plt.title("Calculated size vs Pathology size")
    plt.ylabel("Pathology size (mm)")
    plt.xlabel("Calculated size (mm) from multivariate regression")
    #plt.annotate("R^2 = {:.3f}".format(regr2.score(X, y)), (20,8))
    #plt.annotate("R^2 = {:.3f}".format(regr2.score(X, y)), (20,1))
    plt.annotate("R^2 = {:.3f}".format(regr2.score(X, y)), (5,60))
    #add linear regression line to scatterplot
    #obtain m (slope) and b(intercept) of linear regression line
    #X=g['Calculated size mm image'] #Making it one-dimensional again
    #m, b = np.polyfit(X, y,1)
    #plt.plot(X, m*X+b)
    plt.show()
    
plotting2(g)
#plotting2(gnocancer)


#%%Making subdatasets for the patholgy polyp sizes
gcopy=g.copy()
gcopy.to_csv('gcopy.csv',index=True)
#Polyps under 6 mm
gu6=gcopy[gcopy['Pathology polyp size'] <6]
gu6.to_csv('gu6.csv',index=True)
gu6.to_pickle('gu6.pkl')

#Polyps from 6 to 10mm
g6to10=gcopy[(gcopy['Pathology polyp size'] >=6) & (gcopy['Pathology polyp size'] <10)]
g6to10.to_csv('g6to10.csv',index=True)
g6to10.to_pickle('g6to10.pkl')

#Polyps from 10 to 20mm
g10to20=gcopy[(gcopy['Pathology polyp size'] >=10) & (gcopy['Pathology polyp size'] <20)]
g10to20.to_csv('g10to20.csv',index=True)
g10to20.to_pickle('g10to20.pkl')

#Polyps over 20mm 
go20=gcopy[gcopy['Pathology polyp size'] >=20]
go20.to_csv('go20.csv',index=True)
go20.to_pickle('go20.pkl')



##### MAKING PLOTS FOR THE SUBDATASETS ###
plotting2(gu6)
plotting2(g6to10)
plotting2(g10to20)
plotting2(go20)



#%% Vizualizing errors

#Plotting predicted values over actual values - however the R2 is 1, because we do not have test values but we predict the same values that we trained on

y2=g['Calculated size mm image'].values
visualizer = PredictionError(regr1)
visualizer.fit(X, y1)
visualizer.score(X, y2) 
visualizer.poof()

visualizer = ResidualsPlot(regr1)
visualizer.fit(X, y1)
visualizer.score(X, y2) 
visualizer.poof()

#TEST WITH DIVIDING INTO TRAIN AND TEST SET
X=g[['Image1','Image2','Image3','Image4','Image5']].values
y=g['Pathology polyp size'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

model1 = LinearRegression()
visualizer = PredictionError(model1)
visualizer.fit(X_train, y_train)  
visualizer.score(X_test, y_test)
visualizer.poof()
 

visualizer = ResidualsPlot(model1)
visualizer.fit(X_train, y_train)  
visualizer.score(X_test, y_test)  
visualizer.poof()

def categorise1(row):  
    if row['Calculated size mm image'] <6:
        return '0-5mm'
    elif row['Calculated size mm image'] >= 6 and row['Calculated size mm image'] < 10:
        return '6-9mm'
    elif row['Calculated size mm image'] >= 10  and row['Calculated size mm image'] < 20:
        return '10-19mm'
    elif row['Calculated size mm image'] >= 20:
        return 'Over 20 mm'

g['Predicted category'] = g.apply(lambda row: categorise1(row), axis=1)
#RMSE

#MAKING A CONFUSION MATRIX
#Code from: https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
Xpred1=g['Predicted category']
y_test1=g['Category']
cm = confusion_matrix(y_test1, Xpred1) 

#Naming the labels of the categories in the confusion matrix
cm_df = pd.DataFrame(cm,
                     index = ['0-5mm','10-19mm', '6-9mm', 'Over 20 mm'], 
                     columns = ['0-5mm','10-19mm', '6-9mm', 'Over 20 mm'])

#Plotting the confusion matrix
plt.figure(figsize=(5,4))
sb.heatmap(cm_df, annot=True, cmap='mako')
plt.title('Confusion Matrix for multivariate regression')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test1, Xpred1)))
print('Micro Precision: {:.2f}'.format(precision_score(y_test1, Xpred1, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test1, Xpred1, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test1, Xpred1, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test1, Xpred1, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test1, Xpred1, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test1, Xpred1, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test1, Xpred1, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test1, Xpred1, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test1, Xpred1, average='weighted')))

##Visualizing the errors in mm with histograms     

def visualize_error(sizedataframe, name):
    Xpred=sizedataframe['Calculated size mm image']
    Xpred=Xpred.values.reshape(-1, 1)
    y_test=sizedataframe['Pathology polyp size'].values
    sqrt(mean_squared_error(y_test, Xpred))
    _, ax = plt.subplots()
    
    ax.scatter(x = range(0, y_test.size), y=y_test, c = 'blue', label = 'Actual', alpha = 0.3)
    ax.scatter(x = range(0, Xpred.size), y=Xpred, c = 'red', label = 'Predicted', alpha = 0.3)
    
    plt.title('Actual and predicted values for '+name)
    plt.xlabel('Observations')
    plt.ylabel('mm')
    plt.legend()
    plt.show()
    
    diff = sizedataframe['diff image multi']
    diff.hist(bins = 40)
    plt.title('Histogram of prediction errors for '+ name)
    plt.xlabel('mm prediction error')
    plt.ylabel('Frequency')  
    return print(sqrt(mean_squared_error(y_test, Xpred)))

visualize_error(g, "all data")
visualize_error(gu6, "polyps 0-5mm")
visualize_error(g6to10, "polyps between 6 and 9mm")
visualize_error(g10to20, "polyps between 10 and 19mm")
visualize_error(go20, "polyps 20mm and over")

#FOR LINEAR REGRESSION OF IMAGE 3
def categorise3mm(row):  
    if row['Calculated size mm image 3'] <6:
        return '0-5mm'
    elif row['Calculated size mm image 3'] >= 6 and row['Calculated size mm image 3'] < 10:
        return '6-9mm'
    elif row['Calculated size mm image 3'] >= 10  and row['Calculated size mm image 3'] < 20:
        return '10-19mm'
    elif row['Calculated size mm image 3'] >= 20:
        return 'Over 20 mm'

g['Predicted category for image 3'] = g.apply(lambda row: categorise3mm(row), axis=1)
#RMSE

#MAKING A CONFUSION MATRIX
#Code from: https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
Xpred3mm=g['Predicted category for image 3']
y_test3mm=g['Category']
cm3mm = confusion_matrix(y_test3mm, Xpred3mm) 

#Naming the labels of the categories in the confusion matrix
cm3mm_df = pd.DataFrame(cm3mm,
                     index = ['0-5mm','10-19mm', '6-9mm', 'Over 20 mm'], 
                     columns = ['0-5mm','10-19mm', '6-9mm', 'Over 20 mm'])

#Plotting the confusion matrix
plt.figure(figsize=(5,4))
sb.heatmap(cm3mm_df, annot=True, cmap='mako')
plt.title('Confusion Matrix for linear regression')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test3mm, Xpred3mm)))
print('Micro Precision: {:.2f}'.format(precision_score(y_test3mm, Xpred3mm, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test3mm, Xpred3mm, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test3mm, Xpred3mm, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test3mm, Xpred3mm, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test3mm, Xpred3mm, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test3mm, Xpred3mm, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test3mm, Xpred3mm, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test3mm, Xpred3mm, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test3mm, Xpred3mm, average='weighted')))

g.to_pickle('final_df.pkl') 
g.mean()
g.median()


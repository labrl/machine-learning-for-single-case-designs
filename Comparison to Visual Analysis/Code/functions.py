# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:35:40 2020

@authors: Marc Lanovaz
"""
#Import packages and functions
import numpy as np
import matplotlib.pyplot as plt
from math import tan, radians
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

lm = LinearRegression()
standard_scaler = preprocessing.StandardScaler()

#Set random seed for replicability
np.random.seed(48151)

#Create one series
def create_ABseries(a, t, constant, nb_pointsA, nb_pointsB, SMD):  
    
    nb_points = nb_pointsA + nb_pointsB
    
    #Start with empty series
    data_series = []
    
    #For number of points generate errors
    for i in range(nb_points):
        
        #To deal with first point only
        if not data_series:
            error = np.random.normal()
            data_series.append(error)
        
        #Points other than first - Add autocorrelation
        else: 
            error = a*(data_series[i-1])+np.random.normal()
            data_series.append(error)
    
    #Add trend
    middle_point = np.median(range(nb_points))

    for i in range(nb_points):
        diff = i - middle_point
        data_series[i] = data_series[i] + diff*tan(radians(t))
    
    #Add constant  
    data_series = [x+constant for x in data_series]
    
    #Data labels A and B
    data_labels = ['A'] *nb_pointsA +['B']*nb_pointsB
    
    #Add SMD to each point of phase B
    for j in range(nb_pointsA,nb_points):
        data_series[j] = data_series[j]+ SMD
    
    final_series = np.vstack((data_labels, data_series))
    
    return final_series

#Graph AB and save to pdf
def ABgraph(series, pdfname):
    A = np.where(series[0] == 'A')
    B = np.where(series[0] == 'B')
    valuesA = series[1][A].astype(float)
    valuesB = series[1][B].astype(float)
    
    ylim_value = np.min([0,np.min(valuesA)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(A[0]+1, valuesA, 'k', B[0]+1, valuesB, 'k', marker = 's', 
             clip_on=False)
    plt.axvline(x=len(A[0])+0.5, color = 'k')
    plt.xlabel('Session')
    plt.ylabel('Behavior')
    plt.ylim(ylim_value, np.max(series[1].astype(float))*1.2)
    labels = [item.get_text() for item in ax.get_yticklabels()]

    empty_string_labels = ['']*len(labels)
    ax.set_yticklabels(empty_string_labels)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pdfname.savefig()

#Standardize data so that each graph has a mean of 0 and standard deviation of 1
def standardize_data(x):
    
    features = []
        
    indexA = np.where(x[0]== 'A')
    indexB = np.where(x[0]== 'B')
        
    phaseA = (x[1,indexA].astype(float)).flatten()
    phaseB = (x[1,indexB].astype(float)).flatten()
        
    overallMean = np.mean(np.hstack((phaseA, phaseB)))
    overallSd = np.std(np.hstack((phaseA, phaseB)))
        
    phaseA = (phaseA-overallMean)/overallSd
    phaseA[np.isnan(phaseA)] = 0
    
    phaseB = (phaseB-overallMean)/overallSd
    phaseB[np.isnan(phaseB)] = 0
    
    pointsA = len(phaseA)
    pointsTotal = pointsA + len(phaseB)
    
    features.append([np.mean(phaseA), np.mean(phaseB)])
    features.append([np.std(phaseA),np.std(phaseB)])
    
    lm1 = LinearRegression().fit(np.array(range(pointsA)).reshape(-1,1), 
                    np.expand_dims(phaseA, axis =1))
    features.append([float(lm1.intercept_), float(lm1.coef_)])
    lm2 = LinearRegression().fit(np.array(range(pointsA,pointsTotal))\
                    .reshape(-1,1), np.expand_dims(phaseB, axis =1))
    features.append([float(lm2.intercept_), float(lm2.coef_)])
    
    features = np.array(features).flatten()
    
    return features

#Apply conservative dual-criteria method
f = open("FisheretalList.txt","r")
FisheretalStr = f.read()

def CDCfunction(data, pointsA, pointsB):
    
    #Split data
    A = data[0:pointsA]
    B = data[pointsA:(pointsA+pointsB)]
    
    #Mean line
    meanLine = np.mean(A)+np.std(A)*0.25
    
    #Trend line
    y = np.expand_dims(A, axis=1)
    X = np.array([range(pointsA)]).T
    lm = LinearRegression().fit(X, y)
    trendLine = lm.coef_*np.array(range(pointsA,(pointsA+pointsB))) +lm.intercept_
    trendLine = np.round(trendLine, 3) + np.std(A)*0.25
    
    #Number of points above
    sigPoints = np.sum(np.logical_and(B > meanLine, B > trendLine))
    
    sig_values =  eval(FisheretalStr)
    
    
    if sigPoints >= sig_values[str(pointsB)] :
        return 1
    else:
        return 0
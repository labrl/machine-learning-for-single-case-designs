# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:00:35 2020

@author: Marc Lanovaz
"""
#Import packages and function
import numpy as np
import pandas as pd
import pickle
from functions import CDCfunction

#Load datata
f = open("dataset.txt","rb")
dataset = pickle.load(f)

#Apply CDC Method to each graph
CDCresults = np.empty((0,))

for i in range(len(dataset)):
    series = dataset[i][0]
    pointsA = np.sum(series[0]=='A')
    pointsB = np.sum(series[0]=='B')
    CDCresults = np.hstack((CDCresults, 
                            CDCfunction(series[1].astype(np.float32), 
                            pointsA, pointsB)))
    
#Write results to .csv file
CDCresults = pd.DataFrame(CDCresults)
CDCresults.to_csv('cdc_values.csv', header = False, index=False)

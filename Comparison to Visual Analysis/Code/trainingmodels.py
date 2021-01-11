# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:41:26 2020

@author: Marc Lanovaz
"""
#Import packages and functions
import pickle
import pandas as pd
import numpy as np
import joblib
from functions import create_ABseries, standardize_data

#Set random seed for replicability
np.random.seed(78937)

#Create 96,000 data series for training models
a_values = [0,0.2]
t_values = [0,30]
constant_values = [4,10]
pointsa_values = [3,5]
pointsb_values = [5,10]
smd_values = [0,0,0,0,0,1,2,3,4,5]

dataset = []
for i in range(300):
    for a in a_values: 
        for t in t_values: 
            for constant in constant_values:
                for points_a in pointsa_values:
                    for points_b in pointsb_values:
                        for smd in smd_values: 
                            dataseries = create_ABseries(a, t, constant, points_a,
                                                       points_b, smd)
                            if smd == 0: 
                                dataset.append([dataseries, 0])
                            else:
                                dataset.append([dataseries, 1])

#Ranodmize order of data
shuffled_order = np.random.choice(range(96000), 96000, replace = False)
shuffled_dataset = []

for i in shuffled_order:
    shuffled_dataset.append(dataset[i])

#Organize and standardize training data (96,000 graphs)
x = np.empty((0,8))
y = np.empty((0,))

for i in range(len(shuffled_dataset)):
    series = shuffled_dataset[i][0]
    features = standardize_data(series).reshape(1,-1)
    x = np.vstack((x, features))
    y = np.hstack((y, shuffled_dataset[i][1]))

#Train SGD with code from Lanovaz et al. (2020)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

x_train, x_valid ,y_train, y_valid =\
    train_test_split(x, y, test_size = 0.50, random_state = 48151)

def trainSGD(x_train, y_train, x_valid, y_valid): 
    lr = [0.00001, 0.0001, 0.001, 0.01]
    iterations = np.arange(5,1000,5)
    SGDresults = []
    best_acc = 0
    for i in iterations:
        for n in lr:
            sgd = SGDClassifier(loss = "hinge", penalty = 'elasticnet', alpha = n, 
                                max_iter = i, class_weight = {0:1, 1:0.20}, 
                                random_state=48151)
            sgd.fit(x_train, y_train)
            current_acc = sgd.score(x_valid, y_valid)
            SGDresults.append([n, i, current_acc])
            if current_acc > best_acc:
                best_acc = current_acc
                filename = 'best_modelsgd.sav'
                joblib.dump(sgd, filename)
    return joblib.load('best_modelsgd.sav')

sgd  = trainSGD(x_train, y_train, x_valid, y_valid)

#Predictions of SGD on test data (1,024 graphs)

f = open("dataset.txt","rb")
test_data = pickle.load(f)

sgd_results = np.empty((0,))

for i in range(len(test_data)):
    series = test_data[i][0]
    features = standardize_data(series).reshape(1,-1)
    sgd_results = np.hstack((sgd_results, sgd.predict(features).flatten()))    

sgd_results = pd.DataFrame(sgd_results)
sgd_results.to_csv('sgd_values.csv', header = False, index = False) 


#Train SVC with code from Lanovaz et al. (2020)

from sklearn.svm import SVC

def trainSVC(train_x, train_y, valid_x, valid_y):
    svc = SVC(class_weight = {0:1, 1:0.5})
    
    gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    C = [1,10,100]
    SVCresults = []
    best_acc = 0
    for c in C:
        for n in gamma:
            svc.set_params(gamma =n, C = c)
            svc.fit(train_x, train_y)
            current_acc = svc.score(valid_x, valid_y)
            SVCresults.append([n, c, current_acc])
            if current_acc > best_acc:
                best_acc = current_acc
                filename = 'best_modelsvc.sav'
                joblib.dump(svc, filename)
    return joblib.load('best_modelsvc.sav')

svc = trainSVC(x_train, y_train, x_valid, y_valid)

#Prediction of SVC on test data (1,024 graphs)
svc_results = np.empty((0,))

for i in range(len(test_data)):
    series = test_data[i][0]
    features = standardize_data(series).reshape(1,-1)
    svc_results = np.hstack((svc_results, svc.predict(features).flatten()))    

svc_results = pd.DataFrame(svc_results)
svc_results.to_csv('svc_values.csv', header = False, index = False) 

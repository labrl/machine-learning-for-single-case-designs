# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:23:15 2020

@author: Marc Lanovaz
"""
#Import packages and functions
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import cohen_kappa_score, accuracy_score

#Load entire dataset
with open('dataset.txt', 'rb') as file:
    data = pickle.load(file)

#Extract true values
effect_size = np.empty((0,))

for i in range(len(data)):
    effect_size = np.hstack((effect_size, data[i][1][-1]))

true_values = effect_size.copy()
true_values[true_values >=1] = 1

#Load results of all analyses
expertA = (pd.read_excel('ExpertA.xlsx', header = None).values).flatten()
expertB = (pd.read_excel('ExpertB.xlsx', header = None).values).flatten()
expertC = (pd.read_excel('ExpertC.xlsx', header = None).values).flatten()
expertD = (pd.read_excel('ExpertD.xlsx', header = None).values).flatten()
expertE = (pd.read_excel('ExpertE.xlsx', header = None).values).flatten()

cdc_values = (pd.read_csv('cdc_values.csv', header = None).values).flatten()
sgd_values = (pd.read_csv('sgd_values.csv', header = None).values).flatten()
svc_values = (pd.read_csv('svc_values.csv', header = None).values).flatten()

#Organize data in a list
all_data = [true_values, expertA, expertB, expertC, expertD, expertE,
            cdc_values, sgd_values, svc_values]

method = ['True','A', 'B', 'C', 'D', 'E', 'CDC',
           'sgd', 'svc']

#Compute agreement
agreement = np.empty((0,4))
for i in range(len(all_data)):
    for j in range(len(all_data)):                 
        acc = accuracy_score(all_data[i],all_data[j])
        coh = cohen_kappa_score(all_data[i],all_data[j])
        result = [method[i], method[j], acc, coh]
        agreement = np.vstack((agreement, result))
        
#Overall type I error and power
idx_tI = np.where(true_values == 0)[0]
idx_power = np.where(true_values == 1)[0]

errors = np.empty((0,3))
for i in range(len(all_data)):
    typeI_error = np.sum(all_data[i][idx_tI])/len(idx_tI)
    power = np.sum(all_data[i][idx_power])/len(idx_power)
    result = [method[i], typeI_error, power]
    errors = np.vstack((errors,result))

#Effect of Phase A length
phaseA_values = np.empty((0,))

for i in range(len(data)):
    phaseA_values = np.hstack((phaseA_values, data[i][1][3]))

idx_shortA, = np.where(phaseA_values==3)
idx_longA, = np.where(phaseA_values==5)

idx_shortA_tI = list(set(idx_tI).intersection(idx_shortA))
idx_longA_tI = list(set(idx_tI).intersection(idx_longA))
idx_shortA_power = list(set(idx_power).intersection(idx_shortA))
idx_longA_power = list(set(idx_power).intersection(idx_longA))
        
phaseA = np.empty((0,5))
for i in range(len(all_data)):
    typeI_shortA= np.sum(all_data[i][idx_shortA_tI])/len(idx_shortA_tI)
    typeI_longA = np.sum(all_data[i][idx_longA_tI])/len(idx_longA_tI)
    
    power_shortA= np.sum(all_data[i][idx_shortA_power])/len(idx_shortA_power)
    power_longA = np.sum(all_data[i][idx_longA_power])/len(idx_longA_power)
    
    result = [method[i], typeI_shortA, typeI_longA, power_shortA, power_longA]
    phaseA = np.vstack((phaseA,result))

#Effect of Phase B length
phaseB_values = np.empty((0,))

for i in range(len(data)):
    phaseB_values = np.hstack((phaseB_values, data[i][1][4]))

idx_shortB, = np.where(phaseB_values==5)
idx_longB, = np.where(phaseB_values==10)

idx_shortB_tI = list(set(idx_tI).intersection(idx_shortB))
idx_longB_tI = list(set(idx_tI).intersection(idx_longB))
idx_shortB_power = list(set(idx_power).intersection(idx_shortB))
idx_longB_power = list(set(idx_power).intersection(idx_longB))
        
phaseB = np.empty((0,5))
for i in range(len(all_data)):
    typeI_shortB= np.sum(all_data[i][idx_shortB_tI])/len(idx_shortB_tI)
    typeI_longB = np.sum(all_data[i][idx_longB_tI])/len(idx_longB_tI)
    
    power_shortB= np.sum(all_data[i][idx_shortB_power])/len(idx_shortB_power)
    power_longB = np.sum(all_data[i][idx_longB_power])/len(idx_longB_power)
    
    result = [method[i], typeI_shortB, typeI_longB, power_shortB, power_longB]
    phaseB = np.vstack((phaseB,result))

#Effect of autocorrelation
autocorrelation_values = np.empty((0,))

for i in range(len(data)):
    autocorrelation_values = np.hstack((autocorrelation_values, data[i][1][0]))

idx_auto, = np.where(autocorrelation_values==0.2)
idx_noauto, = np.where(autocorrelation_values==0)

idx_auto_tI = list(set(idx_tI).intersection(idx_auto))
idx_noauto_tI = list(set(idx_tI).intersection(idx_noauto))
idx_auto_power = list(set(idx_power).intersection(idx_auto))
idx_noauto_power = list(set(idx_power).intersection(idx_noauto))
        
autocorrelation = np.empty((0,5))
for i in range(len(all_data)):
    typeI_auto = np.sum(all_data[i][idx_auto_tI])/len(idx_auto_tI)
    typeI_noauto = np.sum(all_data[i][idx_noauto_tI])/len(idx_noauto_tI)
    
    power_auto = np.sum(all_data[i][idx_auto_power])/len(idx_auto_power)
    power_noauto = np.sum(all_data[i][idx_noauto_power])/len(idx_noauto_power)
    
    result = [method[i], typeI_auto, typeI_noauto, power_auto, power_noauto]
    autocorrelation = np.vstack((autocorrelation,result))
    
#Effect of trend
trend_values = np.empty((0,))

for i in range(len(data)):
    trend_values = np.hstack((trend_values, data[i][1][1]))

idx_trend, = np.where(trend_values==30)
idx_notrend, = np.where(trend_values==0)

idx_trend_tI = list(set(idx_tI).intersection(idx_trend))
idx_notrend_tI = list(set(idx_tI).intersection(idx_notrend))
idx_trend_power = list(set(idx_power).intersection(idx_trend))
idx_notrend_power = list(set(idx_power).intersection(idx_notrend))
        
trend = np.empty((0,5))
for i in range(len(all_data)):
    typeI_trend = np.sum(all_data[i][idx_trend_tI])/len(idx_trend_tI)
    typeI_notrend = np.sum(all_data[i][idx_notrend_tI])/len(idx_notrend_tI)
    
    power_trend = np.sum(all_data[i][idx_trend_power])/len(idx_trend_power)
    power_notrend = np.sum(all_data[i][idx_notrend_power])/len(idx_notrend_power)
    
    result = [method[i], typeI_notrend,typeI_trend, power_notrend, power_trend]
    trend = np.vstack((trend,result))
    
#Effect of variability
constant_values = np.empty((0,))

for i in range(len(data)):
    constant_values = np.hstack((constant_values, data[i][1][2]))

idx_constant4, = np.where(constant_values==4)
idx_constant10, = np.where(constant_values==10)

idx_constant4_tI = list(set(idx_tI).intersection(idx_constant4))
idx_constant10_tI = list(set(idx_tI).intersection(idx_constant10))
idx_constant4_power = list(set(idx_power).intersection(idx_constant4))
idx_constant10_power = list(set(idx_power).intersection(idx_constant10))
        
constant = np.empty((0,5))
for i in range(len(all_data)):
    typeI_constant4 = np.sum(all_data[i][idx_constant4_tI])/len(idx_constant4_tI)
    typeI_constant10 = np.sum(all_data[i][idx_constant10_tI])/len(idx_constant10_tI)
    
    power_constant4 = np.sum(all_data[i][idx_constant4_power])/len(idx_constant4_power)
    power_constant10 = np.sum(all_data[i][idx_constant10_power])/len(idx_constant10_power)
    
    result = [method[i], typeI_constant10, typeI_constant4, power_constant10, 
              power_constant4]
    constant = np.vstack((constant,result))
    
#Power by effect size
power_by_es = np.empty((0,3))
for i in range(len(all_data)):
    for j in range(1,6):
        idx = np.where(effect_size == j)[0]
        power = np.sum(all_data[i][idx])/len(idx)
        result = [method[i], j, power]
        power_by_es = np.vstack((power_by_es,result))
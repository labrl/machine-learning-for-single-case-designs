# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:25:50 2020

@author: Marc Lanovaz
"""

#Import packages and functions
import pickle
import numpy as np
from functions import create_ABseries, ABgraph

#Set random seed for replicability
np.random.seed(48151)

#Set values of autocorrelation 'a', trend in degrees 't', number of points in
#phase A and phase B, and standardized mean difference 'smd'
a_values = [0,0.2]
t_values = [0,30]
constant_values = [4,10]
pointsa_values = [3,5]
pointsb_values = [5,10]
smd_values = [0,0,0,1,2,3,4,5]

#Generate 1,024 graphs with varying characteristics
dataset = []
for i in range(4):
    for a in a_values: 
        for t in t_values: 
            for constant in constant_values:
                for points_a in pointsa_values:
                    for points_b in pointsb_values:
                        for smd in smd_values: 
                            dataseries = create_ABseries(a, t, constant, 
                                                    points_a, points_b, smd)
                            dataset.append([dataseries, [a,t,
                                        constant, points_a, points_b, smd]])
    
#Randomize order of graphs
shuffled_order = np.random.choice(range(1024), 1024, replace = False)
shuffled_dataset = []
for i in shuffled_order:
    shuffled_dataset.append(dataset[i])

#Save dataset as dataset.txt
with open("dataset.txt", "wb") as fp:
    pickle.dump(shuffled_dataset, fp)

#Save all graphs in a single pdf file
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('graphsforanalysis.pdf')
for i in range(len(shuffled_dataset)):
    ABgraph(shuffled_dataset[i][0], pp)
pp.close()

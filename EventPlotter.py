#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:50:58 2018

@author: rasmus
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import Reader as rdr
import shifter as shf
import pandas as pd
#import seaborn as sns; sns.set()

#start the timing
start=time.time()

#--------------------------------------
#--------------------------------------
#load the data into a dataframe
#A = rdr.load_events('wave0.txt')
#process it and save in dataframe
#B = shf.shifter(A)
B=pd.read_hdf('test.h5')

#plot heatmap of shifted signals
Map=np.zeros((280,200))
for n in range (0,len(B)):
    for i in range (0,len(B.Samples[n][B.LeftCrossing[n]:B.RightCrossing[n]])):
        if B.Samples[n][i]>0:
            Map[i][int(B.Samples[n][i])]+=1
Map=np.fliplr(Map)
Map=np.rot90(Map, k=1, axes=(0,1))
plt.imshow(Map,cmap='gnuplot', interpolation='nearest',origin = 'lower')
plt.colorbar()
plt.show()



#--------------------------------------
#--------------------------------------

#End time
end = time.time()
print('runtime = ', end-start)

# coding: utf-8
import tof
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
import dask.dataframe as dd
import time
from dask.diagnostics import ProgressBar
#D = tof.cook_data('data/2019-01-28/15sec/test15sec3.txt', threshold=20, maxamp=920, Nchannel=3, Ychannel=4, outpath='data/2019-01-28/15sec.pq', lg=500, sg=60)
print('pulling into memory')
t0 = time.time()
D = tof.load_dataframe('data/2019-01-28/15sec.pq', in_memory=True, mode='full')
print('loaded in %d'%(time.time()-t0), ' seconds')

D_wobbly = D.query('wobbly_baseline==True').reset_index()
D_cfd_late = D.query('cfd_too_late_lg==True').reset_index()
D_cfd_early = D.query('cfd_too_early==True and cfd_trig_rise>=0').reset_index()
D_cfd_fail = D.query('cfd_too_early==True and cfd_trig_rise<0').reset_index()                             
D_cutoff = D.query('cutoff==True').reset_index()

colorlist=['g', 'b', 'orange']
ax1=plt.subplot(2,2,1)
for i in range(0,len(colorlist)):                                                                           
    plt.plot(D_wobbly.samples[i], c=colorlist[i], alpha=0.5)                          
plt.title('Baseline standard > 2 ADC channels %s events'%len(D_wobbly))
plt.xlabel('t(ns)')        
#ax2=plt.subplot(2,2,2)
#for i in range(0,len(colorlist)):
#    plt.plot(D_cfd_early.samples[i], c=colorlist[i], alpha=0.5)
#plt.title('Cfd trigger inside baseline determination window')
#plt.xlabel('t(ns)') 

ax2=plt.subplot(2,2,2)
for i in range(0,len(colorlist)):                              
    plt.plot(D_cfd_fail.samples[i], c=colorlist[i], alpha=0.5)
plt.title('Cfd algorithm failed: %s events'%len(D_cfd_fail))  
plt.xlabel('t(ns)')

ax3=plt.subplot(2,2,3)
for i in range(0,len(colorlist)): 
    plt.plot(D_cfd_late.samples[i], c=colorlist[i], alpha=0.5)
plt.title('Cfd trigger inside longgate integration window: %s events'%len(D_cfd_late))
plt.xlabel('t(ns)') 

ax4=plt.subplot(2,2,4)
for i in range(0,len(colorlist)): 
    plt.plot(D_cutoff.samples[i], c=colorlist[i], alpha=0.5)
plt.title('Overflow events: %s events'%len(D_cutoff))
plt.xlabel('t(ns)') 
plt.show()

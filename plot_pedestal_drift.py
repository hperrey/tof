# coding: utf-8
# coding: utf-8
import tof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import sys
import dask.dataframe as dd
sys.path.append('../analog_tof/')
import pyTagAnalysis as pta

A = pta.load_data('data/analog/Data1631_cooked.root')

ax1=plt.subplot(2, 2, 1)
plt.hist(A.qdc_det0, bins=5000, range=(0,5000), log=True, histtype='step', alpha=0.75, lw=1.5)
plt.title('QDC spectrum: Entire spectrum')
plt.ylabel('counts')
plt.xlabel('qdc bin')

ax2=plt.subplot(2, 2, 2)
dummy = A.query('0<qdc_det0<450')
plt.hist(dummy.qdc_det0, bins=500, range=(0,500), log=False, histtype='step', alpha=0.75, lw=1.5)
plt.ylabel('counts')
plt.xlabel('qdc bin')
plt.title('QDC spectrum: Pedestal closeup')

ax3=plt.subplot(2, 2, 3)
dummy = A.query('0<qdc_det0<5000')
chunksize = 10000
L = len(dummy)
nchunks = int(L/chunksize)
P = [0]*nchunks
C = [0]*nchunks
bins=50
for i in range(0, nchunks):
    H = np.histogram(dummy.qdc_det0[i*chunksize: (i+1)*chunksize], bins=bins, range=(0,500))
    k = np.argmax(H[0])
    P[i] = (H[1][k] + H[1][k+1])/2
    C[i] = np.mean(dummy[i*chunksize: (i+1)*chunksize].query('0<qdc_det0<500').qdc_det0)

for i in range(0, nchunks):
    if i == np.argmin(C):
        plt.hist(dummy.qdc_det0[i*chunksize: (i+1)*chunksize], bins=bins, range=(0,500), log=False, histtype='step', alpha=0.75, lw=1.5, label='chunk %d\nlowest pedestal mean bin'%i)
    if i == np.argmax(C):
        plt.hist(dummy.qdc_det0[i*chunksize: (i+1)*chunksize], bins=bins, range=(0,500), log=False, histtype='step', alpha=0.75, lw=1.5, label='chunk %d\nhighest pedestal mean bin'%i)
plt.title('Pedestal closeup, chunksize = %d'%chunksize)
plt.ylabel('counts')
plt.xlabel('qdc bin')
plt.legend()

ax4 = plt.subplot(2,2,4)
plt.ylabel('mean qdc bin')
plt.legend(loc='upper left')

plt.plot(C, label=('mean bin'), color='g')
plt.xlabel('Chunk number')
plt.ylabel('\nmean qdc bin')
plt.title('mean bin for each chunk')
plt.show()

ax5 = plt.subplot(2,1,1)                                                               
plt.hist(A.qdc_det0, bins=5000, range=(-500,5000), log=True, histtype='step', alpha=0.75, lw=1.5)
plt.title('QDC spectrum')   
plt.ylabel('counts')                         
plt.xlabel('qdc bin')

qdc = [0]*len(A)
for i in range(0, len(C)*chunksize):
    shift = C[i//chunksize]
    qdc[i] = A.qdc_det0[i] - shift
a6 = plt.subplot(2,1,2)
plt.hist(qdc, bins=5000, range=(-500,5000), log=True, histtype='step', alpha=0.75, lw=1.5)
plt.title('QDC spectrum, compensating for pedestal drift in chunks of size %d'%chunksize)
plt.ylabel('counts')                         
plt.xlabel('qdc bin')
plt.show()

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

#D = dd.read_parquet('data/2019-02-13/data1hour.parquet/', engine='pyarrow')
#D = D.drop('samples', axis=1)
#D.to_parquet('data/2019-02-13/data1hour_reduced.parquet/', engine='pyarrow')
D = pd.read_parquet('data/2019-02-13/data1hour_reduced.parquet/', engine='pyarrow')
A = pta.load_data('data/2019-02-13/Data1575_cooked.root')

thrList=[10, 45, 70]
#===QDC Spectra===
#Digitized
ax1=plt.subplot(2, 2, 1)
for i in thrList:
    dummy = D.query('channel==0 and amplitude>%d'%i)
    plt.hist(dummy.query('amplitude>%d'%i).qdc_lg, bins=500, range=(-100000,2250000), log=True, histtype='step', alpha=0.75, lw=1.5, label='Digitized, thr = %d, \n%d events'%(i, len(dummy)))
plt.legend()
plt.title('Long gate pulse integration for varying pulse height thresholds')
plt.xlabel('QDC bin, uncalibrated')
plt.ylabel('counts')
#Analog
ax2=plt.subplot(2, 2, 2)
dummy = A.query('0<qdc_det0<5000')
plt.hist(A.qdc_det0, bins=500, range=(0,5000), log=True, histtype='step', alpha=0.75, lw=1.5, label='Analog setup, \n%d events'%len(dummy))
plt.legend()
plt.title('Long gate pulse integration')
plt.xlabel('QDC bin, uncalibrated')
plt.ylabel('counts')

#===Time of Flight===
#Digitized
ax3=plt.subplot(2, 2, 3)
dlim=[0.5, 105.5]
for i in thrList:
    #add 500 to avoid roundoff errors
    dummy = D.query('channel == 0 and amplitude>%d and %d<=tof<%d'%(i, dlim[0]*1000, dlim[1]*1000))
    plt.hist((dummy.tof+500)/1000, bins=140, range=(dlim[0], dlim[1]), histtype='step', log=True, alpha=0.75, lw=1.5, label='Digitized, thr = %d, \n%d events'%(i, len(dummy)))
plt.title('ToF spectrum')
plt.xlabel('ToF(ns), uncentered')
plt.ylabel('Counts')
plt.ylim(30, 3000)
plt.legend()
#Analog
ax4=plt.subplot(2, 2, 4)
alim=[380.5, 680.5]
#add 0.5 to avoid roundoff errors
dummy=A.query('%d < tdc_det0_yap0 < %d'%(alim[0], alim[1]))
plt.hist((dummy.tdc_det0_yap0), bins=75, range=((alim[0]), (alim[1])), histtype='step', log=True, alpha=0.75, lw=1.5, label='Analog setup, \n%d events'%(len(dummy)))
plt.title('ToF spectrum')
plt.xlabel('ToF, uncentered and uncalibrated)')
plt.ylabel('Counts')
plt.ylim(30, 3000)
plt.legend()
plt.show()




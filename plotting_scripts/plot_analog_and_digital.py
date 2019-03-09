# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import sys
import dask.dataframe as dd
sys.path.append('../../analog_tof/')
sys.path.append('../tof')
import pyTagAnalysis as pta

#D = dd.read_parquet('data/2019-02-13/data1hour.parquet/', engine='pyarrow')
#D = D.drop('samples', axis=1)
#D.to_parquet('data/2019-02-13/data1hour_reduced.parquet/', engine='pyarrow')
D = pd.read_parquet('../data/finalData/data1hour.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'qdc_lg', 'qdc_sg', 'ps', 'tof']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<920')
A = pta.load_data('../data/finalData/Data1793_cooked.root')

#===QDC Spectra===
#Digitized
ax1=plt.subplot(3, 1, 1)
dcal = np.load('../data/finalData/E_call_digi.npy')
w = [17500/5500]*len(D)
plt.hist(dcal[1]+dcal[0]*D.qdc_lg_fine/1000, weights=w, bins=500, range=(0,15), log=True, histtype='step', alpha=0.75, lw=1.5, label='Digitized %d events'%(len(D)))
#plt.legend()
#plt.title('Long gate pulse integration for varying pulse height thresholds')
plt.xlabel('QDC bin, uncalibrated')
plt.ylabel('counts')
#Analog
#ax2=plt.subplot(2, 2, 2)
acal = np.load('../data/finalData/E_call_analog.npy')
dummy = A.query('0<qdc_det0<5000')
plt.hist(acal[1]+acal[0]*A.qdc_det0, bins=500, range=(0,15), log=True, histtype='step', alpha=0.75, lw=1.5, label='Analog setup, \n%d events'%len(dummy))
plt.legend()
#plt.title('Long gate pulse integration')
plt.xlabel('Energy $MeV_{ee}$')
plt.ylabel('counts')

#===Time of Flight===
#Digitized
ax2=plt.subplot(3, 1, 2)
dlim=[0.5, 105.5]
#add 500 to avoid roundoff errors
dummy = D.query('channel == 0 and %d<=tof<%d'%(dlim[0]*1000, dlim[1]*1000))
plt.hist((dummy.tof+500)/1000, bins=140, range=(dlim[0], dlim[1]), histtype='step', alpha=0.75, lw=1.5, label='Digitized \n%d events'%(len(dummy)))
#plt.title('ToF spectrum')
plt.xlabel('ToF(ns), uncentered')
plt.ylabel('Counts')
#plt.ylim(0, 3000)
plt.legend()
#Analog
ax3=plt.subplot(3, 1, 3)
alim=[320.5, 650.5]
#exclude events in the pedestal (self triggers)
dummy=A.query('500<qdc_det0 and %d < tdc_det0_yap0 < %d'%(alim[0], alim[1]))
plt.hist((dummy.tdc_det0_yap0), bins=140, range=((alim[0]), (alim[1])), histtype='step', alpha=0.75, lw=1.5, label='Analog setup, \n%d events'%(len(dummy)))
#plt.title('ToF spectrum')
plt.xlabel('ToF, uncentered and uncalibrated)')
plt.ylabel('Counts')
#plt.ylim(0, 3000)
plt.legend()
plt.show()

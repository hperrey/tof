# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import sys
import dask.dataframe as dd
sys.path.append('../../analog_tof/')
sys.path.append('../tof')
import pyTagAnalysis as pta

Dthres=40
D = pd.read_parquet('../data/finalData/data1hour_clean.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'qdc_lg', 'qdc_sg', 'ps', 'tof', 'invalid']).query('channel==0 and invalid==False and %s<=amplitude<618'%Dthres)
A = pta.load_data('../data/finalData/Data1793_cooked.root')
A['tof'] = 1000 - A['tdc_det0_yap0']

#figure size
plt.figure(figsize=(8,8))

#===QDC Spectra===
#Digitized
ax1=plt.subplot(2, 2, 1)
dcal = np.load('../data/finalData/E_call_digi.npy')

plt.hist(dcal[1]+dcal[0]*D.qdc_lg_fine/1000, bins=500, range=(0,15), log=True, histtype='step', alpha=0.75, lw=1.5, label='Digitized %d events'%(len(D)))
plt.xlabel('QDC bin, uncalibrated')
plt.ylabel('counts')
#Analog
acal = np.load('../data/finalData/E_call_analog.npy')
dummy = A.query('0<qdc_det0<5000')
plt.hist(acal[1]+acal[0]*A.qdc_det0, bins=500, range=(0,15), log=True, histtype='step', alpha=0.75, lw=1.5, label='Analog setup, \n%d events'%len(dummy))
plt.legend()
plt.xlabel('Energy $MeV_{ee}$', fontsize=12)
plt.ylabel('counts', fontsize=12)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)

def gaus(x, a, x0, sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
def fit_gaus(left, right, df):
    x = np.linspace(left, right, right- left)
    H = np.histogram(df.tof, range=(left, right), bins=(right-left))
    y = H[0]
    popt,pcov = curve_fit(gaus, x, y, p0=[max(y), np.mean(x), 5])
    return popt, pcov

#===Time of Flight===
#Digitized
dlim=[0, 200000]
dummy_D = D.query('%d<=tof<%d'%(dlim[0], dlim[1]))
popt_D, pcov_D = fit_gaus(15000, 35000, dummy_D)
dummy_D['tof'] = (D['tof'] - popt_D[1])/1000
dummy_D = dummy_D.query('-20000<tof<100000')
#analog
Tcal = np.load('../data/finalData/T_cal_analog.npy')
alim=[-1000, 1000]
dummy_A = A.query('qdc_det0>500 and %d<=tof<%d'%(alim[0], alim[1]))
popt_A, pcov_A = fit_gaus(350, 400, dummy_A)
dummy_A['tof'] = (A['tof'] - popt_A[1])*(-Tcal[0])
dummy_A = dummy_A.query('-20<tof<100')
#plot ToF
ax2=plt.subplot(2, 2, 2)
plt.hist(dummy_D.tof + 3.3, range(-20, 100), histtype='step', alpha=0.75, lw=1.5, label='digitized: %s coincidences'%(len(dummy_D)))
plt.hist(dummy_A.tof + 3.3, range(-20, 100), histtype='step', alpha=0.75, lw=1.5, label='Analog: %s coincidences'%(len(dummy_A)))
plt.legend()
plt.xlabel('ToF(ns)', fontsize=12)
plt.ylabel('Counts', fontsize=12)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)

##############################
#Different threshold
##############################

Dthres=140
D = pd.read_parquet('../data/finalData/data1hour_clean.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'qdc_lg', 'qdc_sg', 'ps', 'tof', 'invalid']).query('channel==0 and invalid==False and %s<=amplitude<618'%Dthres)
A = pta.load_data('../data/finalData/Data1793_cooked.root')
A['tof'] = 1000 - A['tdc_det0_yap0']


#===QDC Spectra===
#Digitized
ax3=plt.subplot(2, 2, 3)
dcal = np.load('../data/finalData/E_call_digi.npy')

plt.hist(dcal[1]+dcal[0]*D.qdc_lg_fine/1000, bins=500, range=(0,15), log=True, histtype='step', alpha=0.75, lw=1.5, label='Digitized %d events'%(len(D)))
plt.xlabel('QDC bin, uncalibrated')
plt.ylabel('counts')
#Analog
acal = np.load('../data/finalData/E_call_analog.npy')
dummy = A.query('0<qdc_det0<5000')
plt.hist(acal[1]+acal[0]*A.qdc_det0, bins=500, range=(0,15), log=True, histtype='step', alpha=0.75, lw=1.5, label='Analog setup, \n%d events'%len(dummy))
plt.legend()
plt.xlabel('Energy $MeV_{ee}$', fontsize=12)
plt.ylabel('counts', fontsize=12)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)

def gaus(x, a, x0, sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
def fit_gaus(left, right, df):
    x = np.linspace(left, right, right- left)
    H = np.histogram(df.tof, range=(left, right), bins=(right-left))
    y = H[0]
    popt,pcov = curve_fit(gaus, x, y, p0=[max(y), np.mean(x), 5])
    return popt, pcov

#===Time of Flight===
#Digitized
dlim=[0, 200000]
dummy_D = D.query('%d<=tof<%d'%(dlim[0], dlim[1]))
popt_D, pcov_D = fit_gaus(15000, 35000, dummy_D)
dummy_D['tof'] = (D['tof'] - popt_D[1])/1000
dummy_D = dummy_D.query('-20000<tof<100000')
#analog
Tcal = np.load('../data/finalData/T_cal_analog.npy')
alim=[-1000, 1000]
dummy_A = A.query('qdc_det0>500 and %d<=tof<%d'%(alim[0], alim[1]))
popt_A, pcov_A = fit_gaus(350, 400, dummy_A)
dummy_A['tof'] = (A['tof'] - popt_A[1])*(-Tcal[0])
dummy_A = dummy_A.query('-20<tof<100')
#plot ToF
ax4=plt.subplot(2, 2, 4)
plt.hist(dummy_D.tof + 3.3, range(-20, 100), histtype='step', alpha=0.75, lw=1.5, label='digitized: %s coincidences'%(len(dummy_D)))
plt.hist(dummy_A.tof + 3.3, range(-20, 100), histtype='step', alpha=0.75, lw=1.5, label='Analog: %s coincidences'%(len(dummy_A)))
plt.legend()
plt.xlabel('ToF(ns)', fontsize=12)
plt.ylabel('Counts', fontsize=12)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)


plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/CompareResults/comp.pdf', format='pdf')
plt.show()

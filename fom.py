# coding: utf-8
import matplotlib.pyplot as plt
import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar

import seaborn as sns; sns.set(color_codes=True)
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import sys
sys.path.append('tof')
sys.path.append('../analog_tof')
import pyTagAnalysis as pta
from scipy.signal import convolve
import math


N=pd.read_parquet('data/finalData/data1hour_pedestal.pq/', engine='pyarrow', columns=['qdc_lg_fine', 'qdc_sg_fine', 'amplitude','invalid', 'channel']).query('channel==0 and invalid==False and amplitude>40').reset_index().head(1000000)
#N=pta.load_data('data/finalData/Data1793_cooked.root').head(1000000)
if 'qdc_det0' in N:
    flg=0
    fsg=1.7
    N['qdc_lg'] = N.qdc_det0
    N['ps_new'] = ((flg*500+N.qdc_det0)-(fsg*60+N.qdc_sg_det0))/(flg*500+N.qdc_det0).astype(np.float64)
    Ecal = np.load('data/finalData/E_call_analog.npy')
    N['E'] = Ecal[1] + Ecal[0]*N['qdc_lg']
    dummy=N.query('0<ps_new<1 and 0<E<6 and 500<qdc_det0')
    outpath ='/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/fom/FoM'

else:
    fsg=3500
    flg=230
    N['qdc_lg'] = N['qdc_lg_fine']
    N['qdc_sg'] = N['qdc_sg_fine'] 
    N['ps_new'] = ((flg*500+N['qdc_lg'])-(fsg*60+N['qdc_sg']))/(flg*500+N['qdc_lg']).astype(np.float64)
    Ecal = np.load('data/finalData/E_call_digi.npy')/1000
    N['E'] = Ecal[1] + Ecal[0]*N['qdc_lg']
    dummy=N.query('0<ps_new<1 and 0<E<6')
    outpath ='/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/fom/FoM'

def gaus(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
def fit_gaus(left, right, x0, sigma, df):
    x = np.linspace(left,right,right-left)
    H = np.histogram(df.ps_new, range=(left/100, right/100), bins=(right -left))
    y = H[0]
    print(max(y))
    popt,pcov = curve_fit(gaus, x/100, y, p0=[max(y), x0/100, sigma/100])
    return popt, pcov

kernel = [0]*7
a, x0, sigma = 1, 4, 1
for i in range(0,7):
    kernel[i]=gaus(i+1, a, x0, sigma)
kernel=np.array(kernel)
kernel=kernel/sum(kernel)

bins=100
H=np.histogram(dummy['ps_new'], bins=bins, range=(0,1))
G=convolve(H[0], kernel, method='direct', mode='same')
#G=G/max(G)

x=np.linspace((0)/bins,(bins-1)/bins,bins) 
plt.figure(figsize=(8,8))
#plot smoothed spectrum
ax1=plt.subplot(2,1,1)
plt.plot(x, H[0], label='normal')
plt.plot(x, G,label='smoothed')

#peaks
peaks = np.r_[True, G[1:] > G[:-1]] & np.r_[G[:-1] > G[1:], True]
peaks[G < 0.25*max(G)]=False
Plist=np.where(peaks)[0]
#Plist[0]=Plist[0]/bins
#Plist[1]=Plist[0]/bins
#valleys
valleys = np.r_[True, G[1:] < G[:-1]] & np.r_[G[:-1] < G[1:], True]
valleys[0:Plist[0]]=False
valleys[Plist[1]:-1]=False
Vlist=np.where(valleys)[0]

#plot extreme points
plt.axvline(Plist[0]/bins, lw=1.2, alpha=0.7, color='black')
plt.axvline(Plist[1]/bins, lw=1.2, alpha=0.7, color='black')
plt.axvline(Vlist[0]/bins, label='extreme points', lw=1.2, alpha=0.7, color='black')

#fit gaussian
P1, C1 = fit_gaus(left=0, right=Vlist[0], x0=Plist[0], sigma=0.5, df=dummy)
P2, C1 = fit_gaus(left=Vlist[0], right=100, x0=Plist[1], sigma=0.5, df=dummy)
x=np.linspace(0,(bins-1)/bins,bins*10)

fwhm1 = 2*math.sqrt(2*math.log(2))*P1[2]
fwhm2 = 2*math.sqrt(2*math.log(2))*P2[2]
FoM= (P2[1]-P1[1])/(fwhm1+fwhm2)
print(FoM)
plt.plot(x, gaus(x, P1[0], P1[1], P1[2]), '.', ms=3, label='FWHM = %s'%round(fwhm1, 2))
plt.plot(x, gaus(x, P2[0], P2[1], P2[2]), '.', ms=3, label='FWHM = %s'%round(fwhm2, 2))
plt.title('FoM = %s, lgoffset = %s, sg-offset = %s'%(round(FoM, 2), flg, fsg), fontsize=12)
plt.xlim(0,0.6)
plt.xlabel('Tail/total', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.legend()

ax=plt.subplot(2,1,2)
#use the parameter to generate the psd spectrum
#N = pta.load_data('data/finalData/Data1793_cooked.root')
#N['qdc_lg'] = N.qdc_det0
#N['ps_new'] = ((flg*500+N.qdc_det0)-(fsg*60+N.qdc_sg_det0))/(flg*500+N.qdc_det0).astype(np.float64)
#dummy=N.query('0<ps_new<0.5 and 500<qdc_det0<5000')
dummy=dummy.query('0<ps_new<0.6')
plt.hexbin((Ecal[1] + Ecal[0]*dummy.qdc_lg), dummy.ps_new, cmap='viridis', gridsize=(100,100))
plt.xlabel('MeV$_{ee}$', fontsize=12)
plt.ylabel('Tail/total', fontsize=12)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
plt.colorbar()


plt.savefig(outpath+'.pdf', format='pdf')
plt.show()
#plt.close()


import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar

import seaborn as sns#; sns.set(color_codes=True)
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.signal import convolve


fontsize = 10
D = pd.read_parquet('../data/finalData/data1hour_CNN.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'pred', 'qdc_lg', 'qdc_sg', 'ps', 'tof', 'baseline_std']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<610')
D['qdc_lg'] = D['qdc_lg_fine']
D['qdc_sg'] = D['qdc_sg_fine']
D['ps'] = D['ps_fine']
fsg=25000
flg=3000
D['ps'] = ((flg*500+D['qdc_lg_fine'])-(fsg*60+D['qdc_sg_fine']))/(flg*500+D['qdc_lg_fine']).astype(np.float64)
Dcal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/E_call_digi.npy')
Tshift_D = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_D.npy')
D['E'] = (D.qdc_lg*Dcal[0]+Dcal[1])/1000
D['tof'] = (D['tof'] - Tshift_D[1])/1000 + 3.3



def gaus(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
k=7
kernel = [0]*k
a, x0, sigma = 1, 4, 1
for i in range(0,7):
    kernel[i]=gaus(i+1, a, x0, sigma)
kernel=np.array(kernel)
kernel=kernel/sum(kernel)
H = np.histogram(D.tof, bins =120, range=(-10,110))
H = convolve(H[0], kernel)
plt.figure(figsize=(6.2,2.1))
plt.plot(H, color='black')
plt.box(on=None)
plt.xlim(5,100)
textstr='Sketch of a Time of Flight Spectrum'
plt.text(30, 460, textstr, fontsize= fontsize, verticalalignment='top',bbox=dict(facecolor='none', linewidth=1.5, edgecolor='black', pad=0.5, boxstyle='square'))
ax = plt.gca()
ax.annotate('Neutrons', xy=(55, 150), xytext=(72 ,200), fontsize= fontsize, arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),)
ax.annotate('Gammas', xy=(15, 150), xytext=(30,200), fontsize= fontsize, arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),)
plt.xticks([], [])
plt.yticks([], [])
plt.ylabel('Counts')
plt.xlabel('Time(ns)')
plt.show()


dummy=D.query('0<ps<0.3 and 0<E<5').head(50000)
plt.figure(figsize=(6.2,2.1))
sns.kdeplot(dummy.E, dummy.ps, n_levels=10, cmap='viridis')
plt.box(on=None)
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel('Deposited energy ($MeV_{ee}$)')
plt.ylabel('Pulse shape')
plt.show()

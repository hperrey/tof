# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from scipy.signal import convolve
from scipy import asarray as ar,exp

D=pd.read_parquet('../data/finalData/data10min_clean_dropped_samples_CNNpred.pq.pq/',engine='pyarrow')
N=D.query('pred>0.9 and 0<amplitude<600').reset_index()
Y=D.query('pred<0.1 and 0<amplitude<600').reset_index()
D=0
n=np.array([0]*320).astype(np.float64)
y=np.array([0]*320).astype(np.float64)
L=1000
for i in range(0,L):
    s = int(0.5+N.cfd_trig_rise[i]/1000)
    n += N.samples[i][s-20:s+300]/N.amplitude[i]
    s = int(0.5+Y.cfd_trig_rise[i]/1000)
    y += Y.samples[i][s-20:s+300]/Y.amplitude[i]
#ax1=plt.subplot(2,1,1)
#plt.plot(-n/min(n), color='blue', lw=3)
#plt.plot(-y/min(y), color='red', lw=3)

def gaus(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
kernel = [0]*9
a, x0, sigma = 1, 4, 9
for i in range(0,9):
    kernel[i]=gaus(i+1, a, x0, sigma)
kernel=np.array(kernel)
kernel=kernel/sum(kernel)
#ax2=plt.subplot(2,1,2)
n=convolve(-n/min(n), kernel, method='direct', mode='same')
y=convolve(-y/min(y), kernel, method='direct', mode='same')
plt.figure(figsize=(6.2,3))
plt.plot(n, color='blue', lw=2, linestyle='-', label='Neutron')
plt.plot(y, color='red', lw=2, linestyle='--', label='Gamma')
plt.legend(fontsize=12)
#plt.gca().axes.get_yaxis().set_visible(False)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
plt.xlabel('Time (ns)', fontsize=12)
plt.ylabel('Amplitude (arb. units)', fontsize=12)
plt.tight_layout()
plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/DigitalSetup/pulse_types.pdf', format='pdf')
plt.show()

import matplotlib.pyplot as plt
import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar

import seaborn as sns; sns.set(color_codes=True)
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import sys
sys.path.append('../tof')
sys.path.append('../../analog_tof')
import pyTagAnalysis as pta


N = pta.load_data('../data/finalData/analogTimeCal/Data1801_cooked.root')
minimum, maximum = 0, 5000

def gaus(x, a, x0, sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))

def fit_gaus(left, right, df):
    x = np.linspace(left, right, int(0.5+(right-left)/1 + 1))
    H = np.histogram(df.tdc_det0_yap0, range=(minimum, maximum), bins=(maximum-minimum))
    y = H[0][int(left/1+0.5):int(right/1+0.5)+1]
    popt,pcov = curve_fit(gaus, x, y, p0=[max(y), np.mean(x), 5])
    return popt, pcov


npeaks = 6

delays=[122, 102, 70, 50, 20, 0]

p_l = [None]*npeaks
p_r = [None]*npeaks

p_l[0], p_r[0] = 149, 157
p_l[1], p_r[1] = 219, 228
p_l[2], p_r[2] = 335, 342
p_l[3], p_r[3] = 405, 412
p_l[4], p_r[4] = 514, 520
p_l[5], p_r[5] = 583, 589

popt = [None]*npeaks
pcov = [None]*npeaks

for i in range(0, npeaks):
    print(i)
    popt[i], pcov[i] = fit_gaus(p_l[i], p_r[i], N)

plt.figure(figsize=(6.2,5))
fac=1
# for i in range(0, npeaks):
#     ax = plt.subplot(6,3,i+1+6)
#     x = np.linspace(p_l[i], p_r[i], (p_r[i]-p_l[i])*10)
#     plt.plot(x, fac*gaus(x, popt[i][0], popt[i][1], popt[i][2]), '.', ms=6, zorder=4, label="delay\n%s ns"%delays[i])
#     plt.hist(N.tdc_det0_yap0, range(p_l[i]-20, p_r[i]+20))
#     plt.legend(loc='upper left')
#     plt.ylim(0,2000)
#     ax = plt.gca()
#     ax.tick_params(axis = 'both', which = 'both', labelsize = 12)

ax = plt.subplot(2,1,1)
x = np.linspace(100, 699, 600)
plt.hist(N.tdc_det0_yap0, range(100, 699), label='TDC spectrum')
plt.legend(loc='upper left')
plt.ylim(0,2000)
plt.xlabel('TDC channel', fontsize=12)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)


tdc_bins=[]
for i in range(0, npeaks):
    tdc_bins.append(popt[i][1])
def lin(x, a, b):
    return a*x +b
lin_popt, lin_pcov = curve_fit(lin, tdc_bins, delays, p0=[-0.3, 500])

ax = plt.subplot(2,1,2)
plt.scatter(tdc_bins, delays)
x = np.linspace(100, 699, 600)

plt.plot(x, lin(x, lin_popt[0], lin_popt[1]), label='fit: f(x) = %s$\dfrac{ns}{channel}\cdot$x + %s$ns$'%(round(lin_popt[0], 2), round(lin_popt[1], 2) ) )
plt.xlabel('TDC channel', fontsize=12)
plt.ylabel('delay(ns)', fontsize=12)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
plt.legend()
plt.tight_layout()
#plt.subplots_adjust(wspace=0.2, hspace=0.8)
plt.show()


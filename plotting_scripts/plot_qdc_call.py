import matplotlib.pyplot as plt
import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar

import seaborn as sns; sns.set(color_codes=True)
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

d = pd.read_parquet('data/2019-01-28/10min/frame_with_pedestal_dropped_samples.pq', engine='pyarrow')
N = d.query('channel==3 and 20<peak_index<1000 and amplitude>=20')
N.qdc_lg = N.qdc_lg/100
N_pedestal = d.query('channel==3 and amplitude<3')
N_pedestal.qdc_lg = N_pedestal.qdc_lg/100
d = 0

minimum, maximum = 0, 20000

def gaus(x, a, x0, sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))

def fit_gaus(left, right, df):
    x = np.linspace(left, right, int(0.5+(right-left)/1 + 1))
    H = np.histogram(df.qdc_lg, range=(minimum, maximum), bins=(maximum-minimum))
    y = H[0][int(left/1+0.5):int(right/1+0.5)+1]
    popt,pcov = curve_fit(gaus, x, y, p0=[max(y), np.mean(x), 100])
    return popt, pcov


#p1, p2 = 0, 190
#p3, p4 = 6700, 7830
#p5, p6 = 14000, 15150
p1, p2 = 10, 450
p3, p4 = 3400, 4800
p5, p6 = 7500, 10200
popt_1, pcov_1 = fit_gaus(p1, p2, N_pedestal)
popt_2, pcov_2 = fit_gaus(p3, p4, N)
popt_3, pcov_3 = fit_gaus(p5, p6, N)


for p in range(p2, p1, -1):
    ycheck = gaus(p, popt_1[0], popt_1[1], popt_1[2])
    if ycheck > 0.89*popt_1[0]:
        #to do add linear interpolation
        p89_1 = p
        break
for p in range(p4, p3, -1):
    ycheck = gaus(p, popt_2[0], popt_2[1], popt_2[2])
    if ycheck > 0.89*popt_2[0]:
        #to do add linear interpolation
        p89_2 = p
        break
for p in range(p6, p5, -1):
    ycheck = gaus(p, popt_3[0], popt_3[1], popt_3[2])
    if ycheck > 0.89*popt_3[0]:
        #to do add linear interpolation
        p89_3 = p
        break

#Plotting
ax1 = plt.subplot(2, 2, 1)
l1 = minimum; l2 = maximum; b =  int((maximum-minimum)/100); fac = (l2-l1)/b
plt.hist(N.qdc_lg, bins=b, range=(l1,l2), histtype='step', lw=1, log=False, label='QDC spectrum', zorder=1)
plt.hist(N_pedestal.qdc_lg, bins=b, range=(l1,l2), histtype='step', lw=1, log=True, label='QDC spectrum: pedestal', zorder=2)
x = np.linspace(p1, p2, (p2-p1)*10)
plt.plot(x, fac*gaus(x, popt_1[0], popt_1[1], popt_1[2]), '.', ms=3, zorder=3)
x = np.linspace(p3, p4, (p4-p3)*10)
plt.plot(x, fac*gaus(x, popt_2[0], popt_2[1], popt_2[2]), '.', ms=3, zorder=4)
x = np.linspace(p5, p6, (p6-p5)*10)
plt.plot(x, fac*gaus(x, popt_3[0], popt_3[1], popt_3[2]), '.', ms=3, zorder=5)
#plt.scatter([p89_1, p89_2, p89_3], [fac*0.89*popt_1[0], fac*0.89*popt_2[0], fac*0.89*popt_3[0]], s=50, marker='+', color='black', label='89% of peak', zorder=6)
plt.legend(frameon=True)
plt.xlabel('QDC bin')
plt.ylabel('Counts')
#Zoom
ax2=plt.subplot(2, 2, 2)
l1=minimum; l2=maximum; b= int((maximum-minimum)/1); fac=(l2-l1)/b
plt.hist(N_pedestal.qdc_lg, bins=b, range=(l1, l2), histtype='step', lw=1, log=True, color='g', label='QDC spectrum:\nPedestal closeup', zorder=1)
x = np.linspace(p1, p2, 1001)
plt.plot(x, fac*gaus(x, popt_1[0], popt_1[1], popt_1[2]), '.', ms=3, color='r', label='Gaussian fit: pedestal', zorder=2)
#plt.scatter([p89_1, p89_2, p89_3], [fac*0.89*popt_1[0], fac*0.89*popt_2[0], fac*0.89*popt_3[0]], s=50, marker='+', color='black', label='89% of peak', zorder=3)
plt.legend(frameon=True)
plt.xlim(0,1000)
plt.xlabel('QDC bin')
plt.ylabel('Counts')

#fit a line through the two known energies
ax3=plt.subplot(2, 2, 3)
E1 = 2.23
E2 = 4.44
EeMax1 = 2*E1**2/(0.511+2*E1)
EeMax2 = 2*E2**2/(0.511+2*E2)
Elist = [0, EeMax1, EeMax2]
qdclist = [popt_1[1], p89_2, p89_3]
def lin(x, a, b):
    return a*x +b
popt,pcov = curve_fit(lin, qdclist, Elist, p0=[1, 0])
x = np.linspace(minimum, maximum, (maximum-minimum))
plt.plot(x, lin(x, popt[0], popt[1]), label='Calibration fit')
plt.legend(frameon=True)
plt.xlabel('QDC bin')
plt.ylabel('MeV$_{ee}$')
plt.scatter(qdclist, Elist, marker='+', color='black')


#plt.hist(popt[1]+P.qdc_det0*popt[0], bins=1000, range=(0,5), histtype='step', lw=1, label='P')
ax4=plt.subplot(2, 2, 4)
plt.hist(popt[1]+N.qdc_lg*popt[0], bins= int((maximum-minimum)/200), range=(0,((popt[1]+ (maximum-minimum)*popt[0]))), histtype='step', log=True, lw=1, label='Calibrated energy spectrum')
plt.xlabel('MeV$_{ee}$')
plt.ylabel('Counts')
plt.legend(frameon=True)
#plt.xticks(np.arange(0, 8, 1))
plt.show()



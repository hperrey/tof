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


N = pd.read_parquet('../data/finalData/data1hour_clean.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'tof', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'qdc_lg', 'qdc_sg', 'ps']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<614')
C  = pd.read_parquet('../data/finalData/cobalt60_5min.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'qdc_lg', 'qdc_sg', 'ps']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<920')

#N = d.query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<920')
N.qdc_lg_fine = N.qdc_lg_fine/1000
#C = p.query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<920')
C.qdc_lg_fine = C.qdc_lg_fine/1000


minimum, maximum = 0, 30000

def gaus(x, a, x0, sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))

def fit_gaus(left, right, df):
    x = np.linspace(left, right, int(0.5+(right-left)/1 + 1))
    H = np.histogram(df.qdc_lg_fine, range=(minimum, maximum), bins=(maximum-minimum))
    y = H[0][int(left/1+0.5):int(right/1+0.5)+1]
    popt,pcov = curve_fit(gaus, x, y, p0=[max(y), np.mean(x), 100])
    return popt, pcov


p1, p2 = 3000, 5000
p3, p4 = 6700, 8100
p5, p6 = 13000, 15500
#p7, p8 = 42000, 45000

popt_1, pcov_1 = fit_gaus(p1, p2, C)
popt_2, pcov_2 = fit_gaus(p3, p4, N)
popt_3, pcov_3 = fit_gaus(p5, p6, N)
#popt_4, pcov_4 = fit_gaus(p7, p8, N)


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
# for p in range(p8, p7, -1):
#     ycheck = gaus(p, popt_4[0], popt_4[1], popt_4[2])
#     if ycheck > 0.89*popt_4[0]:
#         #to do add linear interpolation
#         p89_4 = p
#         break

#Plotting
plt.figure(figsize=(6.2,8))


ax1 = plt.subplot(3, 1, 1)
l1 = minimum; l2 = maximum
b =  int((maximum-minimum)/100)
fac = (l2-l1)/b
plt.hist(N.qdc_lg_fine, bins=b, range=(l1,l2), histtype='step', lw=1, log=True, zorder=1)

x = np.linspace(p3, p4, (p4-p3)*1)
plt.plot(x, fac*gaus(x, popt_2[0], popt_2[1], popt_2[2]), '.', ms=6, zorder=4, label='2.23 MeV')
x = np.linspace(p5, p6, (p6-p5)*1)
plt.plot(x, fac*gaus(x, popt_3[0], popt_3[1], popt_3[2]), '.', ms=6, zorder=5, label='4.44 MeV')
#plt.scatter([p89_1, p89_2, p89_3], [fac*0.89*popt_1[0], fac*0.89*popt_2[0], fac*0.89*popt_3[0]], s=50, marker='+', color='black', label='89% of peak', zorder=6)
plt.hist(C.qdc_lg_fine, bins=b, range=(l1,l2), histtype='step', lw=1, log=True, zorder=2)
x = np.linspace(p1, p2, (p2-p1)*1)
plt.plot(x, fac*gaus(x, popt_1[0], popt_1[1], popt_1[2]), '.', ms=6, zorder=3, label='1.17 MeV/1.33 MeV')
#x = np.linspace(p7, p8, (p6-p5)*10)
#plt.plot(x, fac*gaus(x, popt_4[0], popt_4[1], popt_4[2]), '.', ms=3, zorder=3)

plt.legend()
plt.xlabel('QDC bin', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.ylim(0.1, 10**5)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)




#fit a line through the two known energies
ax2=plt.subplot(3, 1, 2)
E1 = (1.17+1.33)/2
E2 = 2.23
E3 = 4.44
#E4 = 6.128
EeMax1 = 2*E1**2/(0.511+2*E1)
EeMax2 = 2*E2**2/(0.511+2*E2)
EeMax3 = 2*E3**2/(0.511+2*E3)
#EeMax4 = 2*E4**2/(0.511+2*E4)
Elist = [0, EeMax1, EeMax2, EeMax3]#, EeMax4]
qdclist = [0, p89_1, p89_2, p89_3]#, p89_4]
def lin(x, a, b):
    return a*x +b
popt,pcov = curve_fit(lin, qdclist, Elist, p0=[1, 0])
dev = np.sqrt(np.diag(pcov))
x = np.linspace(minimum, maximum, (maximum-minimum))
plt.plot(x, lin(x, popt[0], popt[1]), label='f(x) = ax+b\n$\sigma_a$ = %s  $MeV_{ee}/QDCbin$\n$\sigma_b$ = %s $MeV_{ee}$'%(round(dev[0],8), round(dev[1], 3)))
plt.legend(frameon=True)
plt.xlabel('QDC bin', fontsize=12)
plt.ylabel('MeV$_{ee}$', fontsize=12)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
plt.scatter(qdclist, Elist, marker='+', color='black')


#plt.hist(popt[1]+P.qdc_det0*popt[0], bins=1000, range=(0,5), histtype='step', lw=1, label='P')
ax3=plt.subplot(3, 1, 3)
plt.hist(popt[1]+N.qdc_lg_fine*popt[0], bins= b, range=(0,((popt[1]+ (maximum-minimum)*popt[0]))), histtype='step', log=True, lw=1, label='Calibrated energy spectrum')
plt.xlabel('MeV$_{ee}$', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.ylim(0.1, 10**5)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/Ecall.pdf', format='pdf')

#plt.xticks(np.arange(0, 8, 1))
plt.show()



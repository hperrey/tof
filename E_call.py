# coding: utf-8
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

M=8000
step=5
B=int(M/step)
E_peak = 0.6617
Cs=pd.read_hdf('data/E_call/Cs_cooked.h5').query('0<longgate<8000')
PuBe = pd.read_hdf('data/2018-10-23/N_cooked.h5').query('0<longgate<8000')#.query('species==0')
H_Cs = np.histogram(Cs.longgate, range=(0,M), bins=B)
############################################################
############################################################
x=np.linspace(20,H_Cs[1][200]+sum(H_Cs[1][0:2])/2,200)#H_Cs[20:101]+sum(H_Cs[1][0:2])/2
y=H_Cs[0][0:200]

n = len(x)
mean = sum(x*y)/(n)
sigma = sum(y*(x-mean)**2)/(n)
mean=1250
sigma=3550


def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,x,y,p0=[max(y),mean,sigma])

#plt.plot(x,y,'b+:',label='data')
#plt.plot(x,gaus(x,popt[0], popt[1], -popt[2]),'r.',label='fit')
#plt.legend()
#plt.show()
############################################################
############################################################
#peakvalue=int(round(popt[1]))#np.argmax(H_Cs[0])
peakIndex = popt[1]-np.log(popt[0]*2*(popt[2]**2)/popt[0])
peakIndex = int(peakIndex/step)


Bins_Cs = H_Cs[1]/(H_Cs[1][peakIndex])*E_peak
H_PuBe = np.histogram(PuBe.longgate, range=(0,M), bins=B)
Bins_PuBe = H_PuBe[1]/(H_Cs[1][peakIndex])*E_peak
stepsize = np.sum(Bins_Cs[0:2])/2
plt.semilogy(Bins_Cs[0:-1]+stepsize, H_Cs[0], ls='steps', label='Cs')
plt.semilogy(Bins_PuBe[0:-1]+stepsize, H_PuBe[0], ls='steps', label='PuBe')

plt.axvline(x=E_peak, ls='--', label='0.6617 MeV')

#plt.plot(x,y,'b+:',label='data')
plt.plot(Bins_Cs[0:200],gaus(x,popt[0], popt[1], -popt[2]),'r.',label='fit')

plt.ylim(10,100000)
plt.title('QDC callibration')
plt.ylabel('counts')
plt.xlabel('$MeV_{ee}$')
plt.legend()
plt.show()

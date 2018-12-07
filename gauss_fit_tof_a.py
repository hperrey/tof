# coding: utf-8
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import sys
sys.path.append('../analog_tof/')
import pyTagAnalysis as pta                  

def GammaCenter(N, LimLeft=400, LimRight=3000, ViewLeft=400, ViewRight=1000, mean=[675,690,690,705], sigma=20):
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    #DeltaL=bins to use in np.hist. higher bins used for background subt
    DeltaL=LimRight-LimLeft
    #number of bins to work with for plotting
    DeltaV=ViewRight-ViewLeft
    #Create the histogram
    H0=np.histogram(N.tdc_det0_yap0, bins=DeltaL, range=(LimLeft,LimRight))
    H1=np.histogram(N.tdc_det0_yap1, bins=DeltaL, range=(LimLeft,LimRight))    
    H2=np.histogram(N.tdc_det0_yap2, bins=DeltaL, range=(LimLeft,LimRight))
    H3=np.histogram(N.tdc_det0_yap3, bins=DeltaL, range=(LimLeft,LimRight))
    #calculate the background
    B0 = int(round(sum(H0[0][1500:2000])/500))
    B1 = int(round(sum(H0[0][1500:2000])/500)) 
    B2 = int(round(sum(H0[0][1500:2000])/500)) 
    B3 = int(round(sum(H0[0][1500:2000])/500)) 
    #define data used for fitting and plotting
    x = np.linspace(ViewLeft+0.5, ViewRight+0.5, DeltaV)
    y0 = H0[0][0:DeltaV]-B0
    y1 = H1[0][0:DeltaV]-B1
    y2 = H2[0][0:DeltaV]-B2
    y3 = H3[0][0:DeltaV]-B3
    #Get correct parameters from scipy
    popt0,pcov0 = curve_fit(gaus, x[200:400], y0[200:400],p0=[max(y0),mean[0], sigma])
    popt1,pcov1 = curve_fit(gaus, x[200:400], y1[200:400],p0=[max(y1),mean[1], sigma])
    popt2,pcov2 = curve_fit(gaus, x[200:400], y2[200:400],p0=[max(y2),mean[2], sigma])
    popt3,pcov3 = curve_fit(gaus, x[200:400], y3[200:400],p0=[max(y3),mean[3], sigma])
    print(popt0[1])
    print(popt1[1])
    print(popt2[1])
    print(popt3[1]) 
    #plot the background subtracted histogram
    plt.plot(x, y0[0: DeltaV], ls='steps', color='red')
    plt.plot(x, y1[0: DeltaV], ls='steps', color='blue')
    plt.plot(x, y2[0: DeltaV], ls='steps', color='green')
    plt.plot(x, y3[0: DeltaV], ls='steps', color='magenta')
    #plot the fit
    xfine=np.linspace(ViewLeft+0.5, ViewRight+0.5, DeltaV*5)
    plt.plot(xfine,gaus(xfine,popt0[0], popt0[1], -popt0[2]),'r.',label='fit yap 0')
    plt.plot(xfine,gaus(xfine,popt1[0], popt1[1], -popt1[2]),'b.',label='fit yap 1')
    plt.plot(xfine,gaus(xfine,popt2[0], popt2[1], -popt2[2]),'g.',label='fit yap 2')
    plt.plot(xfine,gaus(xfine,popt3[0], popt3[1], -popt3[2]),'m.',label='fit yap 3')
    plt.legend()
    plt.show()
    
    tdc_shift_0=np.array(D_a.tdc_det0_yap0, dtype=np.int16)-popt0[1]+300
    tdc_shift_1=np.array(D_a.tdc_det0_yap1, dtype=np.int16)-popt1[1]+300
    tdc_shift_2=np.array(D_a.tdc_det0_yap2, dtype=np.int16)-popt2[1]+300
    tdc_shift_3=np.array(D_a.tdc_det0_yap3, dtype=np.int16)-popt3[1]+300

    plt.hist(tdc_shift_0, bins=1000, range=(-500,500), lw=2, histtype='step', label='yap0 shifted')
    plt.hist(tdc_shift_1, bins=1000, range=(-500,500), lw=2, histtype='step', label='yap1 shifted')
    plt.hist(tdc_shift_2, bins=1000, range=(-500,500), lw=2, histtype='step', label='yap2 shifted')
    plt.hist(tdc_shift_3, bins=1000, range=(-500,500), lw=2, histtype='step', label='yap3 shifted')
    plt.show()
D_a=pta.load_data('../analog_tof/Data1160_cooked.root')
GammaCenter(D_a)

# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp




def GammaCenter(N, LimLeft=20, LimRight=300, ViewLeft=20, ViewRight=120, mean=35, sigma=4):
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    #DeltaL=bins to use in np.hist. higher bins used for background subt
    DeltaL=LimRight-LimLeft
    #number of bins to work with for plotting
    DeltaV=ViewRight-ViewLeft
    #Create the histogram
    H=np.histogram(N.dt, bins=DeltaL, range=(LimLeft,LimRight))
    #calculate the background
    B = int(round(sum(H[0][150:200])/50))
    B=55
    print(B)
    #define data used for fitting and plotting
    x = np.linspace(ViewLeft+0.5, ViewRight+0.5, DeltaV)
    y = H[0][0:DeltaV]-B
    #Get correct parameters from scipy
    popt,pcov = curve_fit(gaus, x[0:50], y[0:50],p0=[max(y),mean, sigma])
    #plot the background subtracted histogram
    plt.plot(x, y[0: DeltaV], ls='steps', color='black')
    #plot the fit
    xfine=np.linspace(ViewLeft+0.5, ViewRight+0.5, DeltaV*5)
    plt.plot(xfine,gaus(xfine,popt[0], popt[1], -popt[2]),'r.',label='fit')
    plt.legend()
    plt.show()

    x=x-popt[1]
    plt.plot(x[35:120], y[35: DeltaV], ls='steps', color='black')
    plt.show()

    x=1/2*(10**9*(1.10/x))**2*1.67*6.24*10**(-9)*10**(-6)
    plt.plot(x[35:120], y[35: DeltaV], ls='steps', color='black')
    plt.xlabel('MeV')
    plt.show()
    print(popt)

D_d=pd.read_hdf('data/2018-10-23/N_cooked.h5')

GammaCenter(D_d)

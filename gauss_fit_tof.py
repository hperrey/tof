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



def GammaCenter(N, LimLeft=400, LimRight=3000, ViewLeft=400, ViewRight=1000, mean=700, sigma=5):
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    #DeltaL=bins to use in np.hist. higher bins used for background subt
    DeltaL=LimRight-LimLeft
    #number of bins to work with for plotting
    DeltaV=ViewRight-ViewLeft
    #Create the histogram
    H=np.histogram(N.tdc_det0_yap0, bins=DeltaL, range=(LimLeft,LimRight))
    #calculate the background
    B = int(round(sum(H[0][1500:2000])/500))
    #define data used for fitting and plotting
    x = np.linspace(ViewLeft+0.5, ViewRight+0.5, DeltaV)
    y = H[0][0:DeltaV]-B
    #Get correct parameters from scipy
    popt,pcov = curve_fit(gaus, x[280:320], y[280:320],p0=[max(y),mean, sigma])
    #plot the background subtracted histogram
    plt.plot(x, y[0: DeltaV], ls='steps', color='black')
    #plot the fit
    xfine=np.linspace(ViewLeft+0.5, ViewRight+0.5, DeltaV*5)
    plt.plot(xfine,gaus(xfine,popt[0], popt[1], -popt[2]),'r.',label='fit')
    plt.legend()
    plt.show()

D_a=pta.load_data('../analog_tof/Data1160_cooked.root')

GammaCenter(D_a)

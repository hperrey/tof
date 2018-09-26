import tof
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#tof spectrum
def ftof(aN=pd.read_hdf('a_neutron.h5'),
         aY=pd.read_hdf('a_gamma.h5'),
         param='area',
         interval=(0,99999)):

    adv_ne213 = aN.query('%d<=%s<%d' %(interval[0], param, interval[1]))
    adv_yap = aY
    print("ToF")
    tof_hist = tof.tof_spectrum(adv_ne213, adv_yap)
    #np.save("tof_hist1", tof_hist)

    x=[0]*len(tof_hist[0])
    for i in range(0,len(x)):
        x[i]=(tof_hist[1][i]+tof_hist[1][i+1])/2
    plt.bar(x, tof_hist[0], width=abs(tof_hist[1][1]-tof_hist[1][0]))
    plt.ylabel('Counts')
    plt.xlabel('time ns')
    plt.show()
    return tof_hist

#Ftof()


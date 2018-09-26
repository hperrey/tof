import tof
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N=pd.read_hdf('testneutron.h5')
aN=tof.adv_framer(N,keep_samples=True)

b=10
for i in range(0,1):
    plt.plot(aN.samples[i])
    plt.plot(aN.edges[i], [0, 0], 'o')
    plt.axvline(aN.refpoint[i], ls='--')
    plt.xlim(aN.edges[i][0]-b, aN.edges[i][1]+b)
    plt.xlabel('time(ns)')
    plt.ylabel('pulseheight V/1024')
    plt.show()

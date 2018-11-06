# coding: utf-8
import sys
sys.path.append('../analog_tof/')
import pyTagAnalysis as pta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
N_d=pd.read_hdf('data/2018-10-23/N_cooked.h5')
N_a=pta.load_data('../analog_tof/Data1160_cooked.root')
N_d=N_d.query('longgate>=2')
Hall_d=np.histogram(N_d.longgate, range=(0,15000), bins=1500)
Hall_a=np.histogram(N_a.qdc_det0, range=(0,15000), bins=1500)
qmax_a=max(Hall_a[0])
qmax_d=max(Hall_d[0])
scaler=qmax_d/qmax_a
qdc_a=scaler*Hall_a[0]
qdc_d=Hall_d[0]
L=15000
Q_d=np.linspace(0,L,1500)
Q_a=np.linspace(0,int(round(L*4146/2266)),1500)
plt.plot(Q_a, qdc_a, ls='steps',lw=2, label='analog')
plt.plot(Q_d, qdc_d, ls='steps',lw=2, label='digital')
plt.legend()
plt.title('scaled analog data to digital data.\nHere the location of the 4.4MeV gamma peak has been used for rescaling analog data.\nEnergy callibration is needed so data can be compared in a proper way. \n')
plt.yscale('log')
plt.ylim(1,3000)
plt.xlim(-50,10000)
plt.show()

# coding: utf-8
import tof
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
tol_right=120
L=2
Hall=[0]*L
H_0=[0]*L
H_1=[0]*L
for i in range(0,L):
    print('\nslice', i+1, '/', L)
    N=pd.read_hdf('data/2018-10-30/N%d.h5'%i)
    Y=pd.read_hdf('data/2018-10-30/Y%d.h5'%i)
    tof.get_gates(N, lg=500, sg=55, offset=10)
    N=N.query('height>20')
    Hall[i]=tof.tof_spectrum(N,Y,tol_right=tol_right)[0]
    H_0[i]=tof.tof_spectrum(N.query('species==0'),Y,tol_right=tol_right)[0]
    H_1[i]=tof.tof_spectrum(N.query('species==1'),Y,tol_right=tol_right)[0]

Hsum_all = Hall[0]
Hsum_filt = H_0[0]+H_1[0]
Hsum_0 = H_0[0]
Hsum_1 = H_1[0]
for i in range(1,L):
    Hsum_all += Hall[i]
    Hsum_filt+=H_0[i]+H_1[i]
    Hsum_0+=H_0[i]
    Hsum_1+=H_1[i]
plt.plot(Hsum_all, ls="steps", color='black') 
plt.plot(Hsum_filt, ls="steps", color='blue')
plt.plot(Hsum_0, ls="steps", color='red')  
plt.plot(Hsum_1, ls="steps", color='green')
plt.ylabel('Counts')
plt.xlabel('$\Delta T$ (ns)')
plt.xlim(10,120)
plt.ylim(0,200)
plt.legend(['Sum_all', 'Hsum_filt', 'Gammas', 'Neutrons'])
plt.title('ToF spectrum, with PSD filtering')
plt.show()

import tof
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Y=pd.read_hdf('data/2018-10-15/yapfront/yapfront_no_samples.h5')
#N=pd.read_hdf('data/2018-10-15/ne213/ne213_no_samples_lg500_sg60_offset10.h5')
tol_right=120
L=9
H=[0]*L
H_0=[0]*L
H_1=[0]*L
for i in range(0,L):
    print('\nslice', i+1, '/', L)
    N=pd.read_hdf('data/2018-10-23/N%d.h5'%i)
    Y=pd.read_hdf('data/2018-10-23/Y%d.h5'%i)
    tof.get_gates(N, lg=500, sg=55, offset=10)
    #N=N.query('height>20')
    H[i]=tof.tof_spectrum(N,Y,tol_right=tol_right)[0]
    H_0[i]=tof.tof_spectrum(N.query('species==0'),Y,tol_right=tol_right)[0]
    H_1[i]=tof.tof_spectrum(N.query('species==1'),Y,tol_right=tol_right)[0]

Hsum = H_0[0]+H_1[0]
Hsum_0 = H_0[0]
Hsum_1 = H_1[0]
for i in range(1,L):
    Hsum+=H[i]
    Hsum_0+=H_0[i]
    Hsum_1+=H_1[i]

K=int(tol_right/2)
Gsum=[0]*60
Gsum_0=[0]*60
Gsum_1=[0]*60
for i in range(0, 60):
    Gsum[i]=Hsum[2*i]+Hsum[2*i+1]
    Gsum_0[i]=Hsum_0[2*i]+Hsum_0[2*i+1]
    Gsum_1[i]=Hsum_1[2*i]+Hsum_1[2*i+1]


x=np.linspace(0.5,tol_right-0.5, K)
width=1
plt.bar(x, Gsum, width=2, alpha=0.3, color='white', linewidth=1, edgecolor=['blue']*len(x))
plt.bar(x, Gsum_0, width=2, alpha=0.3, color='red')
plt.bar(x, Gsum_1, width=2, alpha=0.3, color='blue')
plt.ylabel('Counts')
plt.xlabel('$\Delta T$ (ns)')
plt.xlim(10,120)
#plt.ylim(0,200)
plt.legend(['Sum', 'Gammas', 'Neutrons'])
plt.title('ToF spectrum, with PSD filtering')
plt.show()




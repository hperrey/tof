# coding: utf-8
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
M=8000
step=50
peakIndex=500
B=int(M/step)
Cs=pd.read_hdf('data/E_call/Cs_cooked.h5').query('0<longgate<8000')
N=pd.read_hdf('data/2018-10-23/N_cooked.h5').query('0<longgate<8000')
Q=np.linspace(0, M*0.662/peakIndex, B)
PuBeQDC=np.histogram(N.longgate,  bins=B)
CsQDC=np.histogram(Cs.longgate, bins=B)
plt.semilogy(Q, CsQDC[0], ls='steps', label='QDC Cs')
plt.semilogy(Q, PuBeQDC[0], ls='steps', label='QDC PuBe')
plt.axvline(x=0.662, ls='--', label='0.662 MeV')
plt.title('Energy callibration')
plt.ylabel('counts')
plt.xlabel('$MeV_{ee}$')
plt.legend()
plt.show()

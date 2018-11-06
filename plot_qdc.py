# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
N=pd.read_hdf('data/2018-10-23/N_cooked.h5').query('longgate>=2')
plt.hist(N.longgate, range=(0,15000), bins=750, histtype='step', lw=2, log=True)
plt.hist(N.query('species==0').longgate, range=(0,15000), bins=750, histtype='step', lw=2, log=True)
plt.hist(N.query('species==1').longgate, range=(0,15000), bins=750, histtype='step', lw=2, log=True)
plt.hist(N.query('species==-1').longgate, range=(0,15000), bins=750, histtype='step', lw=2, log=True)
plt.title('psd filtered qdc spectra')
plt.ylabel('counts')
plt.xlabel('longgate')
plt.legend(['Sum','Gammas','Neutrons','Rejected'])
plt.show()

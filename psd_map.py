# coding: utf-8
import numpy as np          
import matplotlib.pyplot as plt                                                                     
import pandas as pd                                                                                                          
import tof                                                                                                                
import sys                                                                                                                    
import seaborn as sns; sns.set(color_codes=True)
N=pd.read_hdf('data/2018-10-23/N_cooked.h5')             
N.reset_index(inplace=True, drop=True) 
Y=pd.read_hdf('data/2018-10-23/Y_cooked.h5')
Y.reset_index(inplace=True, drop=True)
Nq=N.query('longgate<20000')
plt.hist(N.longgate, bins=750, range=(0,15000), histtype='step', linewidth=2, log=True, label='Sum')
plt.hist(N.query('species==1').longgate, bins=750, range=(0,15000), histtype='step', linewidth=2, log=True, label='Neutrons')
plt.hist(N.query('species==0').longgate, bins=750, range=(0,15000), histtype='step', linewidth=2, log=True, label='Gamma')
plt.hist(N.query('species==-1').longgate, bins=750, range=(0,15000), histtype='step', linewidth=2, log=True, label='Rejected')
plt.legend()
plt.xlabel('Longgate')
plt.ylabel('Counts')
plt.xlim(0,8000)
plt.title('QDC spectrum with pulse shape discrimination \nDigital setup')
plt.show()

import tof
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm


lg=490
sg=50
# N=[0]*9
# for i in range(0,9):
#     N[i] = pd.read_hdf('data/2018-10-15/ne213/oct15_10min_ch0_Slice%d.h5'%i)
#     tof.get_gates(N[i], lg=lg, sg=sg)
#     N[i] = N[i].drop('samples', axis=1)
#     print(i)
# N=pd.concat(N)
# N.to_hdf('data/2018-10-15/ne213/ne213_no_samples_lg490_sg50.h5','a')
N=pd.read_hdf('data/2018-10-15/ne213/ne213_no_samples_lg490_sg50.h5')
plt.hexbin(N.longgate, (N.longgate-N.shortgate)/N.longgate, cmap='gnuplot', bins=None, gridsize=500, norm=LogNorm(vmin=1, vmax=1000))
plt.colorbar()
plt.title('lg = %d ns, sg = %d ns, PS=$\dfrac{lg-sg}{lg}$'%(lg,sg))
plt.xlabel('Longgate')
plt.ylabel('PS')
plt.show()

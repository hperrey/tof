import tof
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from matplotlib.colors import LogNorm


def psd_mapper(lg=500, sg=55):
    L=9
    N=[0]*L
    for i in range(0,L):
        print('Frame:', i,'/',L,'\n')
        N[i] = pd.read_hdf('data/2018-10-23/N%d.h5'%i)
        tof.get_gates(N[i], lg=lg, sg=sg)
        N[i] = N[i].drop('samples', axis=1)

    N=pd.concat(N)
    N.to_hdf('data/2018-10-23/cooked.h5','a')
    #N=pd.read_hdf('data/2018-10-23/N_cooked.h5')
    #N=N.query('species==0')
    plt.hexbin(N.longgate, (N.longgate-N.shortgate)/N.longgate, cmap='inferno', bins=None, gridsize=500)#, LogNorm(vmin=1, vmax=500))
    plt.colorbar()
    plt.clim(0,150)
    plt.title('$t_{lg}$=%d ns, $t_{sg}$=%d ns, PS=$\dfrac{lg-sg}{lg}$\n'%(lg,sg))
    plt.xlabel('Longgate')
    plt.ylabel('PS')
    plt.xlim(0,10000)
    plt.ylim(0,1)
    plt.axvline(x=1300, color="white", linestyle='--')
    plt.plot([1300,10000],[0.15,0.24], color="white", linestyle='--')
    #plt.savefig('psd%d.png'%sg)
    #plt.clf()
    plt.show()

import tof
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
#from matplotlib.colors import LogNorm


def psd_mapper(lg=500, sg=48, offset=10, ROT=1):
    L=1
    N=[0]*L
    for i in range(0,L):
        print('Frame:', i,'/',L,'\n')
        N[i] = pd.read_hdf('data/2018-10-23/N%d.h5'%i)
        tof.get_gates(N[i], lg=lg, sg=sg, offset=offset)
        N[i] = N[i].drop('samples', axis=1)

    N=pd.concat(N)
    N.to_hdf('data/2018-10-23/cooked_sg%d_lg%d_offset%d_ROT%f.h5'%(sg,lg,offset,ROT),'a')
    #N=pd.read_hdf('data/2018-10-23/N_cooked.h5')
    N=N.query('longgate<=15000')
    g=sns.jointplot(x=N["longgate"], y=N["ps"], kind='kde', \
                    color='g', ratio=3, n_levels=16, cbar=True, xlim=(-50,6000), ylim=(-0.05,0.4))
    g.ax_joint.legend_.remove()

    #plt.title('$t_{lg}$=%d ns, $t_{sg}$=%d ns, ROT=%f, PS=$\dfrac{lg-ROT\cdot sg}{lg}$\n'%(lg,sg, ROT))
    #plt.xlabel('Longgate')
    #plt.ylabel('PS')
    #plt.savefig('psd_sg%d_ROT%f.png'%(sg, ROT))
    #plt.clf()
    plt.show()

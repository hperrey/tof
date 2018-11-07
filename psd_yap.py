# coding: utf-8
import tof                           
import pandas as pd                   
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
import seaborn as sns; sns.set(color_codes=True)                       
Y=pd.read_hdf('data/2018-10-23/Y1.h5')
tof.get_gates(Y, lg=800)
Yq=Y.query('0<longgate<20000 and shortgate<=longgate and 0<pulsetail/longgate<1')
plt.hexbin(Yq.longgate, Yq.pulsetail/Yq.longgate, gridsize=500, vmin=0)
#, norm=LogNorm(vmin=1, vmax=275))
plt.colorbar()
plt.xlabel('longgate')
plt.ylabel('$ps_{dynamic}$=$\\frac{lg-sg}{lg}$')
plt.title('YAP dynamic pulseshape vs longgate integration')
plt.show()

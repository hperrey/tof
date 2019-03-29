import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar




d = dd.read_parquet('../data/finalData/data1hour_pedestal.pq', engine='pyarrow')
d = d.query('amplitude>40 and channel==0 and invalid==False and 0<ps<1').reset_index()
d=d.head(50)

plt.figure(figsize=(6.2,3.1))
fig = plt.gcf()
#fig.suptitle("Example of digitized pulses with cfd trigger points", fontsize=16)

colorlist=['g', 'b', 'orange', 'purple']

for i in range(0,len(colorlist)):

    trigpoint=int(d.cfd_trig_rise[i]/1000)
    trigpoint_fine = d.cfd_trig_rise[i]/1000 - trigpoint
    print(trigpoint_fine)
    p1 = -20
    p2 = 150
    t=np.linspace(p1, p2-1, p2-p1)

    start=int(d.cfd_trig_rise[i]/1000)+p1
    stop=int(d.cfd_trig_rise[i]/1000)+p2

    plt.plot(t- trigpoint_fine, d.samples[i][start:stop].astype(np.float64)*1000/1023, c=colorlist[i], alpha=0.5, label='amplitude = %s mV'%int(0.5 + d.amplitude[i].astype(np.float64)*1000/1023))
    plt.scatter(t- trigpoint_fine, d.samples[i][start:stop].astype(np.float64)*1000/1023, s=0.7, color=colorlist[i])
    #plt.title(, fontsize=12)
    plt.ylabel('mV', fontsize=10)
    plt.xlabel('t(ns)', fontsize=10)
    #plt.ylim(-75, 50)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 10)
#textstr = "Example of digitized pulses \nwith cfd trigger points"
#plt.text(110, -50, textstr, fontsize=10, verticalalignment='top',bbox=dict(facecolor='None', edgecolor='Black', pad=0.5, boxstyle='square'))
plt.axvline(0, alpha=1, color='black', lw=0.8, label='CFD trigger 30%')
plt.legend()
plt.tight_layout()
plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/DigitalSetup/goodevents.pdf', format='pdf')
plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar



D_wobbly = pd.read_parquet('data/finalData/badevents/D_wobbly.pq', engine='pyarrow').query('amplitude>40 and channel==0').reset_index()
D_cfd_late = pd.read_parquet('data/finalData/badevents/D_cfd_late.pq', engine='pyarrow').query('amplitude>40 and channel==0').reset_index()
D_cfd_early = pd.read_parquet('data/finalData/badevents/D_cfd_early.pq', engine='pyarrow').query('amplitude>40 and channel==0').reset_index()
D_cfd_fail = pd.read_parquet('data/finalData/badevents/D_cfd_fail.pq', engine='pyarrow').query('amplitude>40 and channel==0').reset_index()
D_cutoff = pd.read_parquet('data/finalData/badevents/D_cutoff.pq', engine='pyarrow').query('amplitude>40 and channel==0').reset_index()
d = dd.read_parquet('data/finalData/data1hour_pedestal.pq', engine='pyarrow')
d = d.query('amplitude>40 and channel==0')
with ProgressBar():
    L = 2276267#len(d)
    L_acc = 2186676#len(d.query('invalid==False'))
    L_inv = L-L_acc
    L_wob=len(D_wobbly)
    L_late=len(D_cfd_late)
    L_early=len(D_cfd_early)
    L_fail=len(D_cfd_fail)
    L_cutoff=len(D_cutoff)

colorlist=['g', 'b', 'orange']
dflist = [D_wobbly, D_cfd_early, D_cfd_late, D_cfd_fail, D_cutoff]
#Get similar scales
for i in range(0, len(dflist)):
    dflist[i] = dflist[i].query('amplitude<75').reset_index()
titlelist = ['(A) Unstable baseline: %s%%'%round(L_wob/L*100, 2),
             '(B) Cfd trigger in baseline determination window: %s%%'%round(L_early/L*100, 4),
             '(C) Cfd trigger in longgate integration window: %s%%'%round(L_late/L*100, 2),
             '(D) Cfd algorithm failed: %s%%'%round(L_fail/L*100, 5),
             '(E) Pulse amplitude beyond dynamic range of digitizer: %s%%'%round(L_cutoff/L*100, 6)]


plt.figure(figsize=(6.2,8))
fig = plt.gcf()
#fig.suptitle("Examples of rejected events: Of %s events %s%% were rejected"%(L, round(100*L_inv/L, 2)), fontsize=12)

for k in range(0,4):
    ax=plt.subplot(4,1,k+1)
    for i in range(0,len(colorlist)):
        plt.plot(dflist[k].samples[i].astype(np.float64)*1000/1023, c=colorlist[i], alpha=0.5)
        plt.scatter(np.linspace(0, 1203, 1204), dflist[k].samples[i].astype(np.float64)*1000/1023, s=0.7, color=colorlist[i])
    plt.title(titlelist[k], fontsize=12)
    plt.ylabel('mV', fontsize=12)
    plt.xlabel('t(ns)', fontsize=12)
    if k==4:
        plt.ylim(-75, 50)
    else:
        plt.ylim(-75, 50)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
    if k!=3:
        ax.axes.get_xaxis().set_visible(False)
plt.tight_layout()
plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/DigitalSetup/badevents.pdf', format='pdf')
plt.show()

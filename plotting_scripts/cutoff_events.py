# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
d = pd.read_parquet('../data/finalData/data1hour_pedestal.pq', engine='pyarrow', columns=['amplitude', 'channel'])
d = d.query('amplitude>50 and channel==0')
fac=1000/1023
d['amp_mv']=d['amplitude']*fac

plt.figure(figsize=(6.2,3.1))
plt.hist(d.query('amplitude<=610').amplitude, bins=650, range=(0,650), histtype='step', lw=1.5, log=False, label='Within dynamic range')
err = 100*len(d.query('amplitude>610'))/len(d)
plt.hist(d.query('amplitude>610').amplitude, bins=650, range=(0,650), histtype='step', lw=1.5, log=True, label='Out of dynamic range, %s%%'%err)
plt.ylabel('counts', fontsize=10)
plt.xlabel('ADC', fontsize=10)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 10)
plt.tight_layout()
plt.legend()
plt.show()

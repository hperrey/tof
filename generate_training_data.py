import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
import numpy as np

#training set
waveFormLegth=300
D = dd.read_parquet('data/finalData/CNN/CNNtrainingdata.pq/', engine='pyarrow').query('invalid==False and amplitude < 617 and channel==0 and amplitude>40 and 20000<cfd_trig_rise<1000*(1204-%d)'%(300))#.reset_index()
Ecal = np.load('data/finalData/E_call_digi.npy')/1000
D['E'] = Ecal[1] + Ecal[0]*D['qdc_lg_fine']
gammas2 = D.query('E>4.2 and ps<0.15 and (tof > 29000 or tof<0)').head(200, npartitions=75, compute=False)#.reset_index()
gammas2 = gammas2.repartition(npartitions=1)
D=D.query('(0 < tof < 29000) or (53000< tof < 78000)')
D['y'] = 1
D['y'] = D['y'].where((53000<D['tof']) & (D['tof']<78000), 0)
D1, test = D.random_split([0.75, 0.25])

neutrons = D1.query('(53000< tof < 78000)')
gammas1 = D1.query('0 < tof < 29000')#.reset_index()
with  ProgressBar():
    print("Lazy instructions generated: saving to disk")
    #neutrons.to_parquet('data/finalData/CNN/neutrons.pq', engine='pyarrow', compression='snappy')
    gammas1.to_parquet('data/finalData/CNN/gammas1.pq', engine='pyarrow', compression='snappy')
    gammas2.to_parquet('data/finalData/CNN/gammas2.pq', engine='pyarrow', compression='snappy')
    #test.to_parquet('data/finalData/CNN/test.pq', engine='pyarrow', compression='snappy')
    print("Dataframe saved")



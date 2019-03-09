import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar


#training set
waveFormLegth=300
neutrons = dd.read_parquet('data/finalData/PuBe_lead_shielding_5min.pq', engine='pyarrow').query('ps>0.17 and amplitude>40 and 20000<cfd_trig_rise<1000*(1204-%d)'%(waveFormLegth)).reset_index()
gammas = dd.read_parquet('data/finalData/PuBe_with_plug_5min.pq', engine='pyarrow').query('amplitude>40 and 20000<cfd_trig_rise<1000*(1204-%d)'%(waveFormLegth)).reset_index()
test = dd.read_parquet('data/finalData/data10min.pq', engine='pyarrow').query('amplitude>40 and 20000<cfd_trig_rise<1000*(1204-%d) and 0<tof<300000'%(waveFormLegth)).reset_index()

with  ProgressBar():
    neutrons.to_parquet('data/finalData/CNN/neutrons.pq', engine='pyarrow', compression='snappy')
    #gammas.to_parquet('data/finalData/CNN/gammas.pq', engine='pyarrow', compression='snappy')
    #test.to_parquet('data/finalData/CNN/test.pq', engine='pyarrow', compression='snappy')













# A=dd.read_parquet('data/2019-02-13/data1hour.parquet/', engine='pyarrow')
# A = A.query('0<tof<500000')#.reset_index()
# A = A.repartition(npartitions=A.npartitions // 20)
# A.to_parquet('data/2019-02-13/All.pq', engine='pyarrow')

# A=pd.read_parquet('data/2019-02-13/All.pq/', engine='pyarrow')
# A = A.reset_index()
# L=len(A)
# L1=int(L*0.75)
# L2 = int(L*0.25)
# P1 = A.head(L1)
# P2 = A.tail(L2)
# P1.to_parquet('data/2019-02-13/T.pq', engine='pyarrow')
# P2.to_parquet('data/2019-02-13/V.pq', engine='pyarrow')


# # A=dd.read_parquet('data/2019-01-28/30min/frame30min.parquet/', engine='pyarrow')
# # T=A.query('0<tof<500000')#.reset_index()
# # T = T.repartition(npartitions=T.npartitions // 20)
# # T.to_parquet('data/NandYseperated/T.pq', engine='pyarrow')


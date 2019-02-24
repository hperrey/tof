# coding: utf-8
import seaborn as sns; import matplotlib.pyplot as plt; import dask.dataframe as dd
import pandas as pd
A=dd.read_parquet('data/2019-02-13/data1hour.parquet/', engine='pyarrow')
A = A.query('0<tof<500000')#.reset_index()
A = A.repartition(npartitions=A.npartitions // 20)
A.to_parquet('data/2019-02-13/All.pq', engine='pyarrow')

A=pd.read_parquet('data/2019-02-13/All.pq/', engine='pyarrow')
A = A.reset_index()
L=len(A)
L1=int(L*0.75)
L2 = int(L*0.25)
P1 = A.head(L1)
P2 = A.tail(L2)
P1.to_parquet('data/2019-02-13/T.pq', engine='pyarrow')
P2.to_parquet('data/2019-02-13/V.pq', engine='pyarrow')


# A=dd.read_parquet('data/2019-01-28/30min/frame30min.parquet/', engine='pyarrow')
# T=A.query('0<tof<500000')#.reset_index()
# T = T.repartition(npartitions=T.npartitions // 20)
# T.to_parquet('data/NandYseperated/T.pq', engine='pyarrow')


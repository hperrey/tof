# coding: utf-8
import seaborn as sns; import matplotlib.pyplot as plt; import dask.dataframe as dd

A=dd.read_parquet('data/2019-01-28/10min/frame.parquet/', engine='pyarrow')
val = A.query('0<tof<500000')#.reset_index()
val = val.repartition(npartitions=val.npartitions // 20)
val.to_parquet('data/NandYseperated/val.pq', engine='pyarrow')

A=dd.read_parquet('data/2019-01-28/30min/frame30min.parquet/', engine='pyarrow')
T=A.query('0<tof<500000')#.reset_index()
T = T.repartition(npartitions=T.npartitions // 20)
T.to_parquet('data/NandYseperated/T.pq', engine='pyarrow')


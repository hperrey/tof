import Reader as rdr
import baseliner as bl
import cfd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


def pulse2frame(filename):
    tstart=time.time()
    A=rdr.load_events(filename)
    invSignalLeft=[0]*len(A)
    invSignalRight=[0]*len(A)
    LeftCross=[0]*len(A)
    RightCross=[0]*len(A)
    B=[0]*len(A)
    for n in range(0,len(A)):
        if n%1000==0:
            print('Event', n, '/', len(A))
        B[n]=bl.baseliner2(A.Samples[n])
        invSignalLeft[n],invSignalRight[n] = cfd.inv(A.Samples[n])#-B[n])
        LeftCross[n], RightCross[n] = cfd.zerocrosser(invSignalLeft[n],invSignalRight[n])
        Frame=pd.DataFrame({'TimeStamp': A.TimeStamp,
                            'Samples' : A.Samples, 'Baseline':B,
                            'LeftCrossing':LeftCross,
                            'RightCrossing':RightCross,
                            'invSignalLeft':invSignalLeft,
                            'invSignalRight': invSignalRight})
    tstop=time.time()
    print(tstop-tstart)
    return Frame


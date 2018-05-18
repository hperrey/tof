import Reader as rdr
import crosser as crs
import pandas as pd
import time


def pulse2frame(filename):
    tstart=time.time()
    A=rdr.load_events(filename)
    invSignalLeft=0
    invSignalRight=0
    LeftCross=[0]*len(A)
    RightCross=[0]*len(A)
    for n in range(0,len(A)):
        if n%1000==0:
            print('Event', n, '/', len(A))
        invSignalLeft,invSignalRight = crs.inv(A.Samples[n])
        LeftCross[n], RightCross[n] = crs.zerocrosser(invSignalLeft,invSignalRight)
        Frame=pd.DataFrame({'TimeStamp': A.TimeStamp,
                            'Samples' : A.Samples,
                            'Baseline' : A.Baseline,
                            'LeftCrossing':LeftCross,
                            'RightCrossing':RightCross})
    tstop=time.time()
    print('processing time = ',tstop-tstart)
    return Frame


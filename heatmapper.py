import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import sys

def get_map(frame, lowlim=0, uplim=1000,frac=1):
    t0=time.time()
    windowlength=270
    leftside=50
    heatMap=np.zeros((windowlength,uplim))
    nFrames=int(len(frame)*frac)
    for n in range(0, nFrames):
        crossing=frame.refpoint[n]
        if n%10 == 0 and n>0:
            t = time.time()
            ETA = ((t-t0)/(n))*(nFrames-n)
            ETAh = ETA/3600
            ETAm = (ETAh%1)*60
            ETAs = (ETAm%1)*60
            k = 100*n/nFrames
            sys.stdout.write("\rGenerating heatmap %d%%, ETA: %dh %dm %ds" % (k, ETAh, ETAm, ETAs))
            sys.stdout.flush()
        for i in range(0,windowlength):
            if i+crossing-leftside<1029:
                if max(frame.samples[n])<lowlim or max(frame.samples[n]>uplim):
                    continue
                else:
                    heatMap[i][int(frame.samples[n][i+crossing-leftside])] += 1


    heatMap=np.rot90(heatMap, k=-1, axes=(0,1))
    heatMap=np.fliplr(heatMap)
    #We want all the zero valued bins to be depicted as black. It is easier to look at.
    #heat map with Zeroes boosted to 0.01
    heatmapZboost=heatMap[:][:]
    for u in range(0,uplim):
        #print('u = ', u)
        for y in range(0,windowlength):
            #print('y = ', y)
            if heatmapZboost[u][y] < 1:
                heatmapZboost[u][y] = 0.01

    plt.imshow(heatmapZboost[1:uplim],cmap='gnuplot',aspect='auto', norm=LogNorm(vmin=0.01, vmax=15000), interpolation='nearest',origin = 'lower')
    #for i in range(50, 70):
    #    plt.plot(frame.samples[i][frame.refpoint[i]-leftside:frame.refpoint[i]+windowlength-leftside])
   
    plt.xlabel('Time ns')
    plt.ylabel('ADC value')
    clb = plt.colorbar()
    clb.ax.set_title('Counts')
    plt.axvline(50)
    plt.show()
    tf = time.time()
    print('\n runtime = ', tf-t0)
    return heatMap, heatmapZboost

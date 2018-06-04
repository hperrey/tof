import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def get_map(frame, lowlim=0, uplim=1000,frac=1):
    windowlength=270
    leftside=50#how many nanoseconds are on the left side?
    heatMap=np.zeros((windowlength,uplim))
    nFrames=int(len(frame)*frac)
    for n in range(0, nFrames):
        crossing=frame.Crossing[n]
        if n%100==0:
            print('n = ',n,'/',nFrames)
        for i in range(0,windowlength):# len(frame.Samples[n])):
            #print('i = ',i)
            if i+crossing-leftside<1029:
                if max(frame.Samples[n])<lowlim or max(frame.Samples[n]>uplim):
                    continue
                else:#frame.Samples[n][i+crossing-leftside]<uplim:
                    heatMap[i][int(frame.Samples[n][i+crossing-leftside])]+=1

    #Map=np.flipud(Map)
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
    #for i in range(50,70):
    #    plt.plot(frame.Samples[i][frame.Crossing[i]-40:frame.Crossing[i]+windowlength-40])
    plt.xlabel('Time ns')
    plt.ylabel('ADC value')
    clb = plt.colorbar()
    clb.ax.set_title('Counts')
    plt.show()

    return heatMap, heatmapZboost

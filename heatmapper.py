import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, cm
from matplotlib.colors import LogNorm
import matplotlib.colors as colors

def get_map(frame):
    #Map=[[0]*300 for i in range(0,1029)]
    uplim=1000
    windowlength=270#240
    heatMap=np.zeros((windowlength,uplim))
    for n in range(0, len(frame)):
        crossing=frame.Crossing[n]
        if n%1000==0:
            print('n = ',n,'/',len(frame))
        for i in range(0,windowlength):# len(frame.Samples[n])):
            #print('i = ',i)
            if i+crossing<1029:
                if frame.Samples[n][i+crossing-50]<uplim:
                    heatMap[i][int(frame.Samples[n][i+crossing-40])]+=1

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

    plt.imshow(heatmapZboost[1:uplim],cmap='gnuplot', norm=LogNorm(vmin=0.01, vmax=15000), interpolation='nearest',origin = 'lower')
    for i in range(50,70):
        plt.plot(frame.Samples[i][frame.Crossing[i]-40:frame.Crossing[i]+windowlength-40])
    plt.xlabel('Time ns')
    plt.ylabel('ADC value')
    clb = plt.colorbar()
    clb.ax.set_title('Counts')
    plt.show()

    return heatMap, heatmapZboost

import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, cm
from matplotlib.colors import LogNorm

def get_map(frame):
    #Map=[[0]*300 for i in range(0,1029)]
    uplim=400
    Map=np.zeros((100,uplim))
    for n in range(0, len(frame)):
        crossing=frame.Crossing[n]
        if n%1000==0:
            print('n = ',n,'/',len(frame))
        for i in range(0,100):# len(frame.Samples[n])):
            #print('i = ',i)
            if i+crossing<1029:
                if frame.Samples[n][i+crossing-30]<uplim:
                    Map[i][int(frame.Samples[n][i+crossing-30])]+=1

    #Map=np.flipud(Map)
    Map=np.rot90(Map, k=-1, axes=(0,1))
    Map=np.fliplr(Map)
    #Map=Map[350:650][0:500]
    #plt.imshow(Map[1:200],cmap='gnuplot', interpolation='nearest',origin = 'lower')
    #plt.plot(events.Samples[30][events.Crossing[30]-30:events.Crossing[30]+70])
    #plt.plot(events.Samples[550][events.Crossing[550]-30:events.Crossing[550]+70])
    #plt.plot(events.Samples[666][events.Crossing[666]-30:events.Crossing[666]+70])
    #plt.plot(events.Samples[74000][events.Crossing[74000]-30:events.Crossing[74000]+70])
    #plt.plot(events.Samples[3003][events.Crossing[3003]-30:events.Crossing[3003]+70])
    #plt.xlabel('Time ns')
    #plt.ylabel('ADC value')
    #clb = plt.colorbar()
    #clb.ax.set_title('Counts')
    #plt.show()


    f = figure(figsize=(6.2,5.6))
    ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
    axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])
    im = ax.matshow(Map,cmap='gnuplot', norm=LogNorm(vmin=0.01, vmax=15000),origin = 'lower')
    t = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    f.colorbar(im, cax=axcolor, ticks=t, format='$%.2f$')
    f.show()


    return Map

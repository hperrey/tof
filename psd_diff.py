import matplotlib.pyplot as plt
import pandas as pd
dt=-5
N=pd.read_hdf('data/2018-10-23/N0.h5')
n=0;y=0
for ni in range(0,len(N)):
    if 150<N.height[ni]<160 and N.species[ni]==1:
        start=int(round(N.refpoint_rise[ni]/1000))
        plt.plot(range(0,200),N.samples[ni][start-100:start+100]-N.samples[ni][start-(100+dt):start+(100-dt)], color='blue', alpha=0.2)
        plt.scatter(range(0,200),N.samples[ni][start-100:start+100]-N.samples[ni][start-(100+dt):start+(100-dt)], color='blue', s=0.1)        
        n+=1
    if n>=10:
        break
for yi in range(0,len(N)):
    if 150<N.height[yi]<160 and N.species[yi]==0:
        start=int(round(N.refpoint_rise[yi]/1000))
        plt.plot(range(0,200),N.samples[yi][start-100:start+100]-N.samples[yi][start-(100+dt):start+(100-dt)], color='red', alpha=0.4)
        plt.scatter(range(0,200),N.samples[yi][start-100:start+100]-N.samples[yi][start-(100+dt):start+(100-dt)], color='red', s=0.1)
        y+=1
    if y>=10:
        break
plt.show()

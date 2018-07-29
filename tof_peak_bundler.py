import reader as rdr
import advancedreader as adv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import time

#mode = 0 #pandas
mode = 1 #txt
tstart = time.time()

if mode == 0:
    neutron=pd.read_hdf('data/2018-07-27/N25.hdf')
    gamma=pd.read_hdf('data/2018-07-27/G10.hdf')
elif mode == 1:
    Nthreshold = 10
    Gthreshold = 10
    print('Neutron channel')
    simpleNeutron = rdr.load_events('testneutron.txt', Nthreshold)
    neutron = adv.processframe(simpleNeutron)
    neutron.to_hdf('data/2018-07-27/N%d.hdf' %Nthreshold,'A')
    #recover memory
    #simpleNeutron = None
    print('\nGamma channel')
    simpleGamma = rdr.load_events('testgamma.txt', Gthreshold)
    gamma = adv.processframe(simpleGamma)
    gamma.to_hdf('data/2018-07-27/G%d.hdf' %Gthreshold,'A')
    #recover memory
    #simpleGamma = None
print('\nFrames have been loaded')


peak_0 = []
peak_gamma = []
peak_neutron = []

tol = 100
fac = 16
Thist=np.histogram([],2*tol,range=(-tol,tol))
print("generating ToF spectrum:")
ymin=0
for ne in range(0, len(neutron)):
    for y in range(ymin, len(gamma)):
        Delta=(fac*neutron.Timestamp[ne]+neutron.Refpoint[ne])-(fac*gamma.Timestamp[y]+gamma.Refpoint[y])
        if Delta>tol:
            ymin=y
        if -tol < Delta <tol:
            Thist[0][tol+int(Delta)]+=1
            if 1 < Delta < 4:
                peak_0.append([ne, y])
            elif 12 < Delta < 17:
                peak_gamma.append([ne, y])
            elif 40 < Delta < 46:
                peak_neutron.append([ne, y])
        elif Delta<-tol: 
            break
    i = 100*ne/len(neutron)+1
    #if ne%100==0:
    #    print(ne)
    sys.stdout.write("\r%d%%" % i)
    sys.stdout.flush()

#calculate runtime
tstop=time.time()
minutes = (tstop-tstart)/60
if minutes >= 1:
    seconds = (tstop-tstart)/60%int((tstop-tstart)/60)*60
    minutes = int(minutes)
else:
    seconds = minutes*60
    minutes = 0
print('Processing time: ', minutes, 'min', seconds, 's')
print('Generating histogram')

x=[0]*len(Thist[0])
for i in range(0,len(x)):
    x[i]=(Thist[1][i]+Thist[1][i])/2
plt.bar(x, Thist[0], width=abs(Thist[1][1]-Thist[1][0]))
plt.ylabel('Counts')
plt.xlabel('time ns')
plt.show()

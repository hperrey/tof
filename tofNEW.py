import rdrdois as rdr
import advancedreader as adv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import time


tstart=time.time()
#neutron=pd.read_hdf('N_tres25_2018-07-14.hdf')
#gamma=pd.read_hdf('G_tres100_2018-07-14.hdf')
neutron=pd.read_hdf('Neutron25.hdf')
gamma=pd.read_hdf('Gamma25.hdf')
Ntreshold=25
Gtreshold=25
print('Neutron channel')
simpleNeutron=rdr.load_events('N25-G100ch0.txt',Ntreshold)
neutron=adv.processframe(simpleNeutron)
#recover memory
simpleNeutron = None
print('\nGamma channel')
simpleGamma=rdr.load_events('N25-G100ch1.txt',Gtreshold)
gamma=adv.processframe(simpleGamma)
#recover memory
simpleGamma = None
print('Frames have been loaded')




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
        elif Delta<-tol:
            break
    i = 100*ne/len(neutron)+1
    #if ne%100==0:
    #    print(ne)
    sys.stdout.write("\r%d%%" % i)
    sys.stdout.flush()

tstop=time.time()
print('Processing time: ', tstop-tstart)


x=[0]*len(Thist[0])
for i in range(0,len(x)):
    x[i]=(Thist[1][i]+Thist[1][i])/2
plt.bar(x, Thist[0], width=abs(Thist[1][1]-Thist[1][0]))
plt.ylabel('Counts')
plt.xlabel('time ns')
plt.show()

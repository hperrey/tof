import rdrdois as rdr
import advancedreader as adv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np



neutron=pd.read_hdf('data/neutron.h5')
gamma=pd.read_hdf('data/gamma.h5')
#simpleNeutron=rdr.load_events('data/wave0.txt')#('data/ne21#3.txt')
#simpleGamma=rdr.load_events('data/wave1.txt')#('data/yap.txt')
#neutron=adv.processframe(simpleNeutron)
#gamma=adv.processframe(simpleGamma)
print('Frames have been loaded')




tol=700
Thist=np.histogram([],2*tol,range=(-tol,tol))
print("generating ToF spectrum:")
ymin=0
for ne in range(0, len(neutron)):
    for y in range(ymin, len(gamma)):
        Delta=(neutron.Timestamp[ne]*16+neutron.Refpoint[ne])-(gamma.Timestamp[y]*16+gamma.Refpoint[y])
        if Delta>tol:
            ymin=y+1
        if -tol < Delta <tol:
            Thist[0][int(Delta)]+=1
        elif Delta<-tol:
            break
    i = 100*ne/len(neutron)
    if ne%100==0:
        print(ne)
    sys.stdout.write("\r%d%%" % i)
    sys.stdout.flush()

red=20
Thist_red=[0]*int(len(Thist[0])/red)
for i in range(0, len(Thist[0])):
    Thist_red[int(i/red)]+=Thist[0][i]

plt.plot(Thist_red)
plt.show()
plt.plot(Thist[0])
plt.show()


x=[0]*len(This[0])
for i in rang(0,le(x)):
    x[i]=(This[1][i]+This[1][i])/2
pl.bar(x, This[0], widh=abs(Thist[1][1]-Thist[1][0]))

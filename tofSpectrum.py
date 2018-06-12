#import reader as rdr
#import advancedreader as adv
import matplotlib.pyplot as plt
import pandas as pd

#simpleNeutron=rdr.load_events('data/ne213.txt')
#simpleGamma=rdr.load_events('data/yap.txt')
#neutron=adv.processframe(simpleNeutron)
#gamma=adv.processframe(simpleGamma)
neutron=pd.read_hdf('data/ne213jun11th2018.h5')
gamma=pd.read_hdf('data/yapjun11th2018.h5')
print('Frames have been loaded')





Tlist=[]
#nelist=[]
#ylist=[]
#this needs to be speeded up!
for ne in range(0, len(neutron)):
    if ne%10==0:
        print(ne,'/',len(neutron))
    if neutron.Noevent[ne]==False:
        for y in range(0, len(gamma)):
            if gamma.Noevent[ne]==False:
                if -12000 < (neutron.Timestamp[ne]+neutron.Refpoint[ne])-(gamma.Timestamp[y]+gamma.Refpoint[y]) <12000:
                    Tlist.append((neutron.Timestamp[ne]+neutron.Refpoint[ne])-(gamma.Timestamp[y]+gamma.Refpoint[y]))
                    #nelist.append(ne)
                    #ylist.append(y)
plt.hist(Tlist,200)
pl.show()


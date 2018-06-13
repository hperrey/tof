#import reader as rdr
#import advancedreader as adv
import matplotlib.pyplot as plt
import pandas as pd
import sys


#simpleNeutron=rdr.load_events('data/wave0.txt')#('data/ne213.txt')
#simpleGamma=rdr.load_events('data/wave1.txt')#('data/yap.txt')
#neutron=adv.processframe(simpleNeutron)
#gamma=adv.processframe(simpleGamma)
neutron=pd.read_hdf('data/neutron.h5')
gamma=pd.read_hdf('data/gamma.h5')
print('Frames have been loaded')





Tlist=[]
#nelist=[]
#ylist=[]
#this needs to be speeded up!
print("generating ToF spectrum:")
y_min=0
for ne in range(0, len(neutron)):
    if neutron.Noevent[ne]==False:
        for y in range(y_min, len(gamma)):
            if gamma.Noevent[ne]==False:
                if -3000 < (neutron.Timestamp[ne]+neutron.Refpoint[ne])-(gamma.Timestamp[y]+gamma.Refpoint[y]) <3000:
                    Tlist.append((neutron.Timestamp[ne]+neutron.Refpoint[ne])-(gamma.Timestamp[y]+gamma.Refpoint[y]))
                    y=y_min
                    break
                elif (neutron.Timestamp[ne]+neutron.Refpoint[ne])-(gamma.Timestamp[y]+gamma.Refpoint[y])<-3000:
                    #y=y_min
                    break
                    #nelist.append(ne)
                    #ylist.append(y)
    i = 100*ne/len(neutron)
    sys.stdout.write("\r%d%%" % i)
    sys.stdout.flush()




plt.hist(Tlist,200)
plt.show()


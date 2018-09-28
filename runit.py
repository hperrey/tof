import tof
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#read in stuff
N = pd.read_hdf('N10.h5')
Y = pd.read_hdf('Y25.h5')
#N = pd.read_hdf('testneutron.h5')
#Y = pd.read_hdf('testgamma.h5')

#tof spectrum
print("ToF")
print("\ntof0")
tof_hist0 = tof.tof_spectrum(N.query('height<=50'), Y)
np.save("tof_hist0", tof_hist0)
#print("\ntof1")
#tof_hist1 = tof.tof_spectrum(N.query('height>51'), Y)
#np.save("tof_hist1", tof_hist1)


x=[0]*len(tof_hist0[0])
for i in range(0,len(x)):
    x[i]=(tof_hist0[1][i]+tof_hist0[1][i+1])/2

p1 = plt.bar(x, tof_hist0[0], width=abs(tof_hist0[1][1]-tof_hist0[1][0]), alpha=0.65)
#p2 = plt.bar(x, tof_hist1[0], width=abs(tof_hist0[1][1]-tof_hist0[1][0]), bottom=tof_hist0[0], alpha=0.65)
#plt.legend((p1[0], p2[0]), ('height $\leq$ 0.049 V', 'height > 0.05 V'))
#plt.title('Stacked ToF Histograms')
plt.title('ToF Histograms')
plt.legend('NE213, YAP $\Delta$T')

plt.show()

#tof_hist = tof_hist0[0]+tof_hist1[0]+tof_hist2[0]+tof_hist3[0]
#plt.bar(x, tof_hist, width=abs(tof_hist0[1][1]-tof_hist0[1][0]))
#plt.show()

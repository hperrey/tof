import pandas as pd;
import tof
import matplotlib.pyplot as plt
adv_ne213 = pd.read_hdf('test_ne213_10.h5')
adv_yap = pd.read_hdf('test_yap_25.h5')


#tof spectrum
print("ToF")
tof_hist = tof.tof_spectrum(adv_ne213, adv_yap)

x=[0]*len(tof_hist[0])
for i in range(0,len(x)):
    x[i]=(tof_hist[1][i]+tof_hist[1][i+1])/2
plt.bar(x, tof_hist[0], width=abs(tof_hist[1][1]-tof_hist[1][0]))
plt.ylabel('Counts')
plt.xlabel('time ns')
plt.show()

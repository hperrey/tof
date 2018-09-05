import tof
import matplotlib.pyplot as plt
import time
import numpy as np

t0 = time.time()

#ne213 channel
print("neutron")
ne213path="sep5th_set1_ch1"
b_ne213=tof.basic_framer('%s.txt' %ne213path, 30)
b_ne213.to_hdf('b_ne213_4min_27August2018_threshold_10','a')
#b_ne213=tof.basic_framer('testneutron.txt', 30)
adv_ne213=tof.adv_framer(b_ne213, keep_samples=False)
adv_ne213.to_hdf('adv_ne213_4min_27August2018_threshold_10','a')
b_ne213 = None
t1 = time.time()
print('\nne213: ', t1-t0)

#yap channel
print("yap")
yappath="sep5th_set1_ch0"
b_yap=tof.basic_framer('%s.txt'%yappath, 60)
b_yap.to_hdf('b_yap_4min_27August2018_threshold_30','a')
#b_yap=tof.basic_framer('testgamma.txt', 60)
adv_yap=tof.adv_framer(b_yap, keep_samples=False)
adv_yap.to_hdf('adv_yap_4min_27August2018_threshold_30','a')
b_yap = None
t2 = time.time()
print('\nyap: ', t2-t1)

#tof spectrum
print("ToF")
tof_hist = tof.tof_spectrum(adv_ne213, adv_yap)
np.save("tof_hist1", tof_hist)
t3 = time.time()
print('\ntof: ', t3-t2)

x=[0]*len(tof_hist[0])
for i in range(0,len(x)):
    x[i]=(tof_hist[1][i]+tof_hist[1][i+1])/2
plt.bar(x, tof_hist[0], width=abs(tof_hist[1][1]-tof_hist[1][0]))
plt.ylabel('Counts')
plt.xlabel('time ns')
plt.show()

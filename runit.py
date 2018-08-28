import tof
import matplotlib.pyplot as plt
import time

t0 = time.time()

#ne213 channel
print("neutron")
b_ne213=tof.basic_framer('testneutron.txt', 10)
adv_ne213=tof.adv_framer(b_ne213)
#b_ne213 = None
t1 = time.time()
print('\nne213: ', t1-t0)
#yap channel
print("yap")
b_yap=tof.basic_framer('testgamma.txt', 25)
adv_yap=tof.adv_framer(b_yap)
#b_yap = None
t2 = time.time()
print('\nyap: ', t2-t1)

#tof spectrum
print("ToF")
tof_hist = tof.tof_spectrum(adv_ne213, adv_yap)
t3 = time.time()
print('\ntof: ', t3-t2)

plt.plot(tof_hist[0])
plt.show()

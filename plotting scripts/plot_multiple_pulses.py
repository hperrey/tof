import tof; import numpy as np; import matplotlib.pyplot as plt


#Way old. 27-July
plt.subplot(321)
plt.subplots_adjust(hspace=0.4)
axes = plt.gca()
axes.set_ylim([-10,200])
N=tof.basic_framer('testneutron.txt', 150)
aN=tof.adv_framer(N)
for i in range(0, len(N)):
    if 200>np.max(N.samples[i])>100:
        plt.plot(N.samples[i][aN.refpoint[i]-50:aN.refpoint[i]+350], 'g', alpha=0.7)
        plt.title("NE213, 27-Jul")
        break

plt.subplot(322)
plt.subplots_adjust(hspace=0.4)
axes = plt.gca()
axes.set_ylim([-10,200])
Y=tof.basic_framer('testgamma.txt', 150)
aY=tof.adv_framer(Y)
for i in range(0, len(Y)):
    if 200>np.max(Y.samples[i])>100:
        plt.plot(Y.samples[i][aY.refpoint[i]-50:aY.refpoint[i]+350], 'g', alpha=0.7)
        plt.title("YAP,  27-Jul")
        break

#just after terminator being applied
plt.subplot(323)
plt.subplots_adjust(hspace=0.4)
axes = plt.gca()
axes.set_ylim([-10,200])
N0=tof.basic_framer('AfterTerminator31Aug-N10-G10ch0.txt', 150)
aN0=tof.adv_framer(N0)
for i in range(0, len(N0)):
    if 200>np.max(N0.samples[i])>100:
        plt.plot(N0.samples[i][aN0.refpoint[i]-50:aN0.refpoint[i]+350], 'c', alpha=0.7)
        plt.title("NE213, 31-Aug")
        break

plt.subplot(324)
plt.subplots_adjust(hspace=0.4)
axes = plt.gca()
axes.set_ylim([-10,200])
Y0=tof.basic_framer('AfterTerminator31Aug-N10-G10ch1.txt', 150)
aY0=tof.adv_framer(Y0)
for i in range(0, len(Y0)):
    if 200>np.max(Y0.samples[i])>100:
        plt.plot(Y0.samples[i][aY0.refpoint[i]-50:aY0.refpoint[i]+350], 'c', alpha=0.7)
        plt.title("YAP, 31-Aug")
        break

# #The next monday
plt.subplot(325)
plt.subplots_adjust(hspace=0.4)
axes = plt.gca()
axes.set_ylim([-10,200])
N1=tof.basic_framer('AfterTerminator03Sep-N10-G10ch0.txt', 150)
aN1=tof.adv_framer(N1)
for i in range(0, len(N1)):
    if 200>np.max(N1.samples[i])>100:
        plt.plot(N1.samples[i][aN1.refpoint[i]-50:aN1.refpoint[i]+350], 'm', alpha=0.7)
        plt.title("NE213, 03-Sep")
        break

plt.subplot(326)
plt.subplots_adjust(hspace=0.4)
axes = plt.gca()
axes.set_ylim([-10,200])
Y1=tof.basic_framer('AfterTerminator03Sep-N10-G10ch1.txt', 150)
aY1=tof.adv_framer(Y1)
for i in range(0, len(Y1)):
    if 200>np.max(Y1.samples[i])>100:
        plt.plot(Y1.samples[i][aY1.refpoint[i]-50:aY1.refpoint[i]+350], 'm', alpha=0.7)
        plt.title("YAP,  03-Sep")
        break



plt.show()

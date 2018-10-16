import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#N = pd.read_hdf('N10.h5')
N=pd.read_hdf('test0_D0.h5')
L=[0, 10, 40, 90, 200, 400]
P=[0]*(len(L)-1)
check=[0]*(len(L)-1)
for i in range(0, len(N)):
    for u in range(0,len(check)):
        if check[u] == 0 and L[u]<N.height[i]<L[u+1]:
            P[u]=i
            check[u]=1
            break
    if sum(check)==len(check):
        break



#cfd alignment
for i in P:
    F=N.refpoint[i]/1000
    plt.plot(range(0,len(N.samples[i]))-F, N.samples[i]/1024*1000, alpha=0.73)
plt.axvline(x=0, linestyle='--')
plt.xlim(-20,70)
plt.title('CFD alignment of NE213 pulses. CFD fraction = 0.5')
plt.xlabel('ns')
plt.ylabel('mV')
plt.legend(('height=%d mV'%round(N.height[P[0]]*1000/1024),
            'height=%d mV'%round(N.height[P[1]]*1000/1024),
            'height=%d mV'%round(N.height[P[2]]*1000/1024),
            'height=%d mV'%round(N.height[P[3]]*1000/1024),
            'height=%d mV'%round(N.height[P[4]]*1000/1024)))
plt.show()



#leading edge
def get_lead_edge(samples, threshold):
     l_edge=0
     for i in range(0,len(samples)):
          if samples[i]>threshold:
               l_edge=i
               break
     return l_edge
threshold=15
for i in P:
    F=get_lead_edge(N.samples[i], threshold)
    plt.plot(np.array(range(0,len(N.samples[i])))-F, N.samples[i]/1024*1000, alpha=0.73)
plt.axvline(x=0, linestyle='--')
plt.xlim(-20,70)
plt.title('Leading edge alignment of NE213 pulses. Threshold = 15 mV')
plt.xlabel('ns')
plt.ylabel('mV')
plt.legend(('height=%d mV'%round(N.height[P[0]]*1000/1024),
            'height=%d mV'%round(N.height[P[1]]*1000/1024),
            'height=%d mV'%round(N.height[P[2]]*1000/1024),
            'height=%d mV'%round(N.height[P[3]]*1000/1024),
            'height=%d mV'%round(N.height[P[4]]*1000/1024)))
plt.show()     

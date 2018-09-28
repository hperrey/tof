import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;
R=pd.read_hdf('N10.h5')
W=5

#no alignment
a=0
b=0
c=0
d=0
e=0
S=1023
l=0
r=l+S
for i in range(0,len(R)):
     if True: #if R.ch[i]==0:
          if a==0:
               if 40>R.height[i]>0:
                    a+=1
                    plt.plot(range(0,S), R.samples[i][l:r]/1024,c='m', alpha=1)
          elif b==0:
               if 80>R.height[i]>40:
                    b+=1
                    plt.plot(range(0,S), R.samples[i][l:r]/1024,c='m', alpha=0.85)
          elif c==0:
               if 120>R.height[i]>80:
                    c+=1
                    plt.plot(range(0,S), R.samples[i][l:r]/1024,c='m', alpha=0.70)
          elif d==0:
               if 160>R.height[i]>120:
                    d+=1
                    plt.plot(range(0,S), R.samples[i][l:r]/1024,c='m', alpha=0.55)
          elif e==0:
               if 999>R.height[i]>400:
                    e+=1
                    plt.plot(range(0,S), R.samples[i][l:r]/1024,c='m', alpha=0.4)
          elif a==b==c==d==e==1:
               break
plt.xlim((300,950))
plt.xlabel('ns')
plt.ylabel('V')
plt.show()

def cfdplotter(point):
     a=0
     b=0
     c=0
     d=0
     e=0
     S=30
     for i in range(0,len(R)):
          if True: #if R.ch[i]==0:
               if a==0:
                    if 40>R.height[i]>0:
                         a+=1
                         plt.plot(range(0,2*S), R.samples[i][point[i]-S:point[i]+S]/1024,c='g', alpha=1)
               if b==0:
                    if 80>R.height[i]>40:
                         b+=1
                         plt.plot(range(0,2*S), R.samples[i][point[i]-S:point[i]+S]/1024,c='g', alpha=0.85)
               if c==0:
                    if 120>R.height[i]>80:
                         c+=1
                         plt.plot(range(0,2*S), R.samples[i][point[i]-S:point[i]+S]/1024,c='g', alpha=0.70)
               if d==0:
                    if 160>R.height[i]>120:
                         d+=1
                         plt.plot(range(0,2*S), R.samples[i][point[i]-S:point[i]+S]/1024,c='g', alpha=0.55)
               if e==0:
                    if 999>R.height[i]>400:
                         e+=1
                         plt.plot(range(0,2*S), R.samples[i][point[i]-S:point[i]+S]/1024,c='g', alpha=0.40)
               if a==b==c==d==e==1:
                    break
     plt.axvline(x=S, linestyle='--')
     plt.xlabel('ns')
     plt.ylabel('V')
     plt.show()

def peakplotter():
     point=0
     a=0
     b=0
     c=0
     d=0
     e=0
     S=30
     for i in range(0,len(R)):
          if True: #if R.ch[i]==0:
               if a==0:
                    if 40>R.height[i]>0:
                         a+=1
                         point=np.argmax(R.samples[i])
                         plt.plot(range(0,2*S), R.samples[i][point-S:point+S]/1024,c='b', alpha=1)
               if b==0:
                    if 80>R.height[i]>40:
                         b+=1
                         point=np.argmax(R.samples[i])
                         plt.plot(range(0,2*S), R.samples[i][point-S:point+S]/1024,c='b', alpha=0.85)
               if c==0:
                    if 120>R.height[i]>80:
                         c+=1
                         point=np.argmax(R.samples[i])
                         plt.plot(range(0,2*S), R.samples[i][point-S:point+S]/1024,c='b', alpha=0.70)
               if d==0:
                    if 160>R.height[i]>120:
                         d+=1
                         point=np.argmax(R.samples[i])
                         plt.plot(range(0,2*S), R.samples[i][point-S:point+S]/1024,c='b', alpha=0.55)
               if e==0:
                    if 999>R.height[i]>400:
                         e+=1
                         point=np.argmax(R.samples[i])
                         plt.plot(range(0,2*S), R.samples[i][point-S:point+S]/1024,c='b', alpha=0.40)
               if a==b==c==d==e==1:
                    break
     plt.axvline(x=S, linestyle='--')
     plt.xlabel('ns')
     plt.ylabel('V')
     plt.show()

def l_edge_plotter():
     point=0
     a=0
     b=0
     c=0
     d=0
     e=0
     S=30
     threshold=10
     for i in range(0,len(R)):
          if True: #if R.ch[i]==0:
               if a==0:
                    if 40>R.height[i]>0:
                         a+=1
                         point=get_lead_edge(R.samples[i], threshold)
                         plt.plot(range(0,2*S), R.samples[i][point-S:point+S]/1024,c='r', alpha=1)
               if b==0:
                    if 80>R.height[i]>40:
                         b+=1
                         point=get_lead_edge(R.samples[i], threshold)
                         plt.plot(range(0,2*S), R.samples[i][point-S:point+S]/1024,c='r', alpha=0.85)
               if c==0:
                    if 120>R.height[i]>80:
                         c+=1
                         point=get_lead_edge(R.samples[i], threshold)
                         plt.plot(range(0,2*S), R.samples[i][point-S:point+S]/1024,c='r', alpha=0.70)
               if d==0:
                    if 160>R.height[i]>120:
                         d+=1
                         point=get_lead_edge(R.samples[i], threshold)
                         plt.plot(range(0,2*S), R.samples[i][point-S:point+S]/1024,c='r', alpha=0.55)
               if e==0:
                    if 999>R.height[i]>400:
                         e+=1
                         point=get_lead_edge(R.samples[i], threshold)
                         plt.plot(range(0,2*S), R.samples[i][point-S:point+S]/1024,c='r', alpha=0.40)
               if a==b==c==d==e==1:
                    break
     plt.axvline(x=S, linestyle='--')
     plt.xlabel('ns')
     plt.ylabel('V')
     plt.show()

#peak
peakplotter()
#cfd
cfdplotter(R.refpoint)
#leading edge
def get_lead_edge(samples, threshold):
     l_edge=0
     for i in range(0,len(samples)):
          if samples[i]>threshold:
               l_edge=i
               break
     return l_edge
l_edge_plotter()


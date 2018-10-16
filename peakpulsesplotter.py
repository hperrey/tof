import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tof
import sys

def persistantTracePlot(F, filterlist, name, gridsize=100):
    ''' show "persistent" trace picture '''
    fig, ax = plt.subplots()
    time=[0]
    ampl=[0]
    for i in range(0, len(F)):
        if i in filterlist:
            time = np.concatenate([time,range(0,len(F.samples[i]))-F.refpoint[i]/1000])
            ampl = np.concatenate([ampl, np.array(F.samples[i])/1024*1000])
    plt.xlabel('ns')
    plt.ylabel('mV')
    plt.hexbin(time, ampl, bins='log', gridsize=gridsize)
    plt.title("persistent trace: %s"%name)
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    # adjust axis range
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,1.1*y1,1.1*y2))
    return fig, ax

def tof_spectrum(ne213, yap, fac=8, tol_left=0, tol_right=80):
    Neutrons=[0]*len(ne213)
    Gammas=[0]*len(ne213)
    ymin=0
    G_index = 0
    N_index = 0
    tof_hist = np.histogram([], tol_left+tol_right, range=(tol_left, tol_right))
    counter = 0
    for row in ne213.itertuples():
        ne=row[0]
        counter+= 1
        k = 100*counter/len(ne213)
        sys.stdout.write("\rGenerating tof spectrum %d%%" % k)
        sys.stdout.flush()
        for y in range(ymin, len(yap)):
            Delta=int(round(((fac*1000*ne213.timestamp[ne]+ne213.refpoint[ne])-(fac*1000*yap.timestamp[y]+yap.refpoint[y]))/1000))
            if Delta > tol_right:
                ymin = y
            if tol_left < Delta <tol_right:
                tof_hist[0][tol_left+int(Delta)] += 1
                #print(Delta)
                if 12 < Delta < 18:
                    #print('gamma')
                    Gammas[G_index] = ne
                    G_index+=1
                elif 45<Delta<65:
                    #print('neutron')
                    Neutrons[N_index] = ne
                    N_index+=1
            elif Delta < -tol_right:
                break
    Neutrons = Neutrons[0:N_index]
    Gammas = Gammas[0:G_index]    
    return tof_hist, Neutrons, Gammas


#Y=tof.basic_framer('4minutes2018-07-27-N10-G10ch1.txt', 30)
#Y.to_hdf('Y30.h5', 'a')
#N=tof.basic_framer('test_N20NY20_30sch1.txt', 20)
#N.to_hdf('testN20.h5', 'a')
N=pd.read_hdf('data/2018-10-11/ne213/oct11_10min_ch0_Slice2.h5')
#N1=pd.read_hdf('data/2018-10-02/ne213ch0/D1.h5')
#N2=pd.read_hdf('data/2018-10-02/ne213ch0/D2.h5')
#FN=[N0,N1,N2]
#N=pd.concat(FN)
Y=pd.read_hdf('data/2018-10-11/yapfront/oct11_10min_ch1_Slice2.h5')
#Y1=pd.read_hdf('data/2018-10-02/yapch1/D1.h5')
#Y2=pd.read_hdf('data/2018-10-02/yapch1/D1.h5')
#FY=[Y0,Y1,Y2]
#Y=pd.concat(FY)
#N=pd.read_hdf('testneutron10.h5')
#Y=pd.read_hdf('testgamma10.h5')
H, Neutrons, Gammas = tof_spectrum(N,Y)

x=[0]*len(H[0])
for i in range(0,len(x)):
    x[i]=(H[1][i]+H[1][i+1])/2
plt.bar(x, H[0], width=abs(H[1][1] - H[1][0]))
plt.ylabel('Counts')
plt.xlabel('time ns')
plt.show()


max_peak_G = 0
for i in Gammas:
    if N.height[i]>max_peak_G:
        max_peak_G=N.height[i]

max_peak_N = 0
for i in Neutrons:
    if N.height[i] > max_peak_N:
        max_peak_N = N.height[i]

persistantTracePlot(N, Neutrons, gridsize=max_peak_N, name="Neutron peak 45-65 ns")
plt.xlim(-50,400)
plt.show()
persistantTracePlot(N, Gammas, gridsize=max_peak_G, name="Gamma peak 12-18 ns")
plt.xlim(-50,400)
plt.show()

for i in Neutrons:
    F=N.refpoint[i]/1000
    plt.plot(range(0,len(N.samples[i]))-F, N.samples[i]/1024*1000, alpha=0.15,color='blue')
    plt.axvline(x=0, linestyle='--',color='blue')
plt.title('Neutron peak pulses: 45-65 ns')
plt.xlabel('ns')
plt.ylabel('mV')
plt.xlim(-50,400)
plt.show()

for i in Gammas:
    F=N.refpoint[i]/1000
    plt.plot(range(0,len(N.samples[i]))-F, N.samples[i]/1024*1000, alpha=0.15,color='r')
    plt.axvline(x=0, linestyle='--',color='r')
plt.title('Gamma peak pulses: 12-18 ns')
plt.xlabel('ns')
plt.ylabel('mV')
plt.xlim(-50,400)
plt.show()

LN=[]
for index in Neutrons:
    LN.append(N.height[index]/1024*1000)
plt.hist(LN,bins=50,range=(0,300),alpha=0.45)
plt.xlabel('mV')
plt.ylabel('counts')
plt.title('neutron peak pulse height spectrum')
plt.show()

LY=[]
for index in Gammas:
    LY.append(Y.height[index]/1024*1000)
plt.hist(LY,bins=50,range=(0,300),alpha=0.45, color='r')
plt.xlabel('mV')
plt.ylabel('counts')
plt.title('gamma peak pulse height spectrum')
plt.show()

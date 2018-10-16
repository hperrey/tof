import numpy as np
import pandas as pd
import csv
import sys
from itertools import islice
import time
import matplotlib.pyplot as plt


def basic_framer(filename, threshold, frac=0.5, nlines=0, startline=0, nTimeResets=0):
    #Get number of lines
    if nlines == 0:
        nlines=sum(1 for line in (open(filename)))
    nevents = int(nlines/8)
    samples = [None]*nevents
    #nTimeResets=0
    timestamp = np.array([0]*nevents, dtype=np.int64)
    refpoint = np.array([0]*nevents, dtype=np.int32)
    left = np.array([0]*nevents, dtype=np.int16)
    right = np.array([0]*nevents, dtype=np.int16)
    peak_index = np.array([0]*nevents, dtype=np.int16)
    height = np.array([0]*nevents, dtype=np.int16)
    area = np.array([0]*nevents, dtype=np.int16)
    #shortgate = np.array([0]*nevents, dtype=np.int16)
    #longgate = np.array([0]*nevents, dtype=np.int16)
    edges = [None]*nevents
    try:
        with open(filename, newline='\n') as f:
            reader = csv.reader(f)
            line_index = startline
            event_index = 0
            idx=0
            if line_index > 0:
                for row in reader:
                    if idx == startline-1:
                        break
                    else:
                        idx+=1

            #go to the startline
            for row in reader:
                line_index +=1
                #only go through lines belonging to the current block
                if line_index >= startline+nlines:
                    break
                if line_index%8 == 6:
                    k = 100*(line_index-startline)/nlines+1
                    sys.stdout.write("\rGenerating dataframe %d%%" % k)
                    sys.stdout.flush()
                    dummytimestamp = int(row[0].split()[3])
                    if event_index > 0:
                        if dummytimestamp < timestamp[event_index-1]-nTimeResets*2147483647:
                            nTimeResets += 1
                            #print('\ntime reset!\n')
                    timestamp[event_index]= (dummytimestamp+nTimeResets*2147483647)
                if line_index%8 == 0:#every eigth row is the data
                    dummy = row[0].split()
                    dummy=[int(i) for i in dummy]
                    samples[event_index] = np.array(dummy ,dtype=np.int16)
                    #B is the number of samples used to calculate baseline
                    #We don't care about events that have large peaks or noise in this interval
                    B = 20
                    baseline = int(sum(samples[event_index][0:B])/B)
                    samples[event_index] -= baseline
                    #check the polarity and check if the pulse crosses threshold
                    peak_index[event_index] = np.argmax(np.absolute(samples[event_index]))
                    if np.absolute(samples[event_index][peak_index[event_index]]) < threshold:
                        continue
                    elif np.absolute(samples[event_index][peak_index[event_index]]) >= threshold:
                        if samples[event_index][peak_index[event_index]] < 0:
                            samples[event_index] *= -1
                        #get pulse height and pulse edge bins
                        height[event_index] = samples[event_index][peak_index[event_index]]
                        edges[event_index] = find_edges(samples[event_index], peak_index[event_index])
                        left[event_index]=edges[event_index][0]
                        right[event_index]=edges[event_index][1]
                        # if event is contained in samp window,
                        #then get full, short and long gate area + go to nex  event.
                        if edges[event_index][0] < edges[event_index][1]:
                            area[event_index] = np.trapz(samples[event_index][edges[event_index][0]:edges[event_index][1]])
                            refpoint[event_index] = cfd(samples[event_index], frac)
                            if refpoint[event_index] == -1:
                                continue
                            #sg = int((edges[event_index][1]-edges[event_index][0])*0.7)
                            #lg = int((edges[event_index][1]-edges[event_index][0])*0.9)
                            #shortgate[event_index] = np.trapz(samples[event_index][refpoint[event_index]-5:refpoint[event_index]+sg])
                            #shortgate[event_index] = np.trapz(samples[event_index][edges[event_index][0]:edges[event_index][0]+sg])
                            #longgate[event_index] = np.trapz(samples[event_index][refpoint[event_index]-5:refpoint[event_index]+lg])
                            #longgate[event_index] = np.trapz(samples[event_index][edges[event_index][0]:edges[event_index][0]+lg])
                            event_index += 1
        #throw away empty rows.
        samples = samples[0:event_index]
        timestamp = timestamp[0:event_index]
        height = height[0:event_index]
        peak_index = peak_index[0:event_index]
        edges = edges[0:event_index]
        area = area[0:event_index]
        #shortgate = shortgate[0:event_index]
        #longgate = longgate[0:event_index]
        refpoint = refpoint[0:event_index]
        left = left[0:event_index]
        right = right[0:event_index]
    except IOError:
        return None
    return pd.DataFrame({'timestamp': timestamp,
                         'samples' : samples,
                         'height' : height,
                         'peak_index':peak_index,
                         'edges' : edges,
                         'area' : area,
                         #'shortgate' : shortgate,
                         #'longgate' : longgate,
                         'refpoint' : refpoint,
                         'left' : left,
                         'right' : right})

def get_gates(frame, lg=500, sg=60, offset=10):
    longgate=np.array([0]*len(frame), dtype=np.int16)
    shortgate=np.array([0]*len(frame), dtype=np.int16)
    species=np.array([0]*len(frame), dtype=np.int8)
    for i in range(0, len(frame)):
        k = round(100*i/len(frame))
        sys.stdout.write("\rCalculating gates %d%%" % k)
        sys.stdout.flush()
        
        start=int(round(frame.refpoint[i]/1000))-offset
        longgate[i] = np.trapz(frame.samples[i][start:start+lg])
        shortgate[i] = np.trapz(frame.samples[i][start:start+sg])

        if shortgate[i]>longgate[i]:
            #workaround. need to deal with reflections properly!
            longgate[i]=1
            shortgate[i]=1
        if longgate[i]<=0 or shortgate[i]<=0:
            longgate[i]=1
            shortgate[i]=1
        if (longgate[i]-shortgate[i])/longgate[i] < 0.08-longgate[i]*0.05/1000:
            species[i] = -1
        elif (longgate[i]-shortgate[i])/longgate[i] < 0.03+longgate[i]*0.025/1000:
            species[i] = 0
        else:
            species[i] = 1
        frame['species'] = species
        frame['longgate']=longgate
        frame['shortgate']=shortgate
    return 0

def cfd(samples, frac):
    peak = np.max(samples)
    index = 0
    for i in range(0,len(samples)):
        if samples[i] > frac*peak:
            index = i
            break
        else:
            index = 0
    slope = (samples[index] - samples[index-1])#divided by 1ns
    if slope == 0:
        np.save('array',samples)
        print('\nslope == 0!!!!')
        print('\nindex=', index)
        print('\n', samples[index-5:index+5])
        tfine = -1
    else:
        tfine = 1000*(index-1) + int(round(1000*(peak*frac-samples[index-1])/slope))
    return tfine


def find_edges(samples, refpoint):
    edges = [0,0]
    for i in range(refpoint, 0, -1):
        if samples[i]<2:
            edges[0]=i
            break
    for i in range(refpoint, len(samples)):
        if samples[i]<2:
            edges[1]=i
            break
    return edges

def get_frames(filename, threshold, frac=0.5):
    time0 = time.time()
    nlines=sum(1 for line in (open(filename)))
    nlinesBlock = 2**21 # lines per block
    #Number of full blocks
    nBlocks = int(nlines/nlinesBlock)
    #Number of lines in the final block
    nlinesBlockF = (nlines%nlinesBlock)
    Blocklines =[nlinesBlock]*(nBlocks+1)
    Blocklines[-1] = nlinesBlockF
    #we need nBlocks +1 dataframes
    FrameList=[0]*len(Blocklines)
    nTimeResets=0
    for i in range(0, len(Blocklines)):
        #if i>0:
        #    nTimeResets=int((FrameList[i-1].timestamp[len(FrameList[i-1])-1])/2147483647)
        print('\n -------------------- \n frame', i+1, '/', len(FrameList), '\n --------------------')
        #print('---nTimeResets=', nTimeResets,'---')
        FrameList[i] = basic_framer(filename, threshold, frac, nlines=Blocklines[i], startline=i*nlinesBlock,nTimeResets=0)
    time1 = time.time()
    deltaT=time1-time0
    print('Runtime: ', deltaT/60, 'minutes')
    return FrameList



def tof_spectrum(ne213, yap, fac=8, tol_left=0, tol_right=80):
    ymin=0
    tof_hist = np.histogram([], tol_left+tol_right, range=(tol_left, tol_right))
    #tof_hist1 = np.histogram([], tol_left+tol_right, range=(tol_left, tol_right))
    #tof_hist2 = np.histogram([], tol_left+tol_right, range=(tol_left, tol_right))

    #for ne in range(0, len(ne213)):
    counter=0
    for row in ne213.itertuples():
        ne=row[0]
        counter += 1
        k = 100*counter/len(ne213)
        sys.stdout.write("\rGenerating tof spectrum %d%%" % k)
        sys.stdout.flush()
        for y in range(ymin, len(yap)):
            Delta=int(round(((fac*1000*ne213.timestamp[ne]+ne213.refpoint[ne])-(fac*1000*yap.timestamp[y]+yap.refpoint[y]))/1000))
            if Delta > tol_right:
                ymin = y
            if tol_left < Delta <tol_right:
                tof_hist[0][tol_left+int(Delta)] += 1
            elif Delta < -tol_right:
                break
    return tof_hist

#unsuccesful attempt at an improved tof algorithm.
def tof_spectrum2(N, Y, fac=8, tol_left=0, tol_right=100):
    tof_hist = np.histogram([], tol_left+tol_right, range=(tol_left, tol_right))
    counter=0
    for row_N in N.itertuples():
        counter+=1
        percentage =  100*counter/len(N)
        sys.stdout.write("\rGenerating tof spectrum %d%%"%percentage)
        sys.stdout.flush()
        i=row_N[0]
        N_ts=fac*1000*N.timestamp[i]+N.refpoint[i]
        Y_window=Y.query('%s-%s<1000*timestamp<%s+%s'%(N_ts, tol_left, N_ts, tol_right))
        for row_Y in Y_window.itertuples():
            y=row_Y[0]
            DeltaT=int(round(fac*1000*Y_window.timestamp[y]+Y_window.refpoint[y] - N_ts))
            tof_hist[0][DeltaT] += 1
    return tof_hist

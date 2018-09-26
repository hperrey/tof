import numpy as np
import pandas as pd
import csv
import sys

def basic_framer(filename, threshold, frac=0.5):
    #Get number of lines
    nlines=sum(1 for line in (open(filename)))
    nevents = int(nlines/8)
    nTimeResets = 0
    samples = [None]*nevents
    timestamp = np.array([0]*nevents, dtype=np.int64)
    refpoint = np.array([0]*nevents, dtype=np.int16)
    left = np.array([0]*nevents, dtype=np.int16)
    right = np.array([0]*nevents, dtype=np.int16)
    peak_index = np.array([0]*nevents, dtype=np.int16)
    height = np.array([0]*nevents, dtype=np.int16)
    area = np.array([0]*nevents, dtype=np.int16)
    shortgate = np.array([0]*nevents, dtype=np.int16)
    longgate = np.array([0]*nevents, dtype=np.int16)
    edges = [None]*nevents
    try:
        with open(filename, newline='\n') as f:
            reader = csv.reader(f)
            line_index = 0
            event_index = 0
            for row in reader:
                line_index +=1
                if line_index%8 == 6:
                    k = 100*line_index/nlines+1
                    sys.stdout.write("\rGenerating basic dataframe %d%%" % k)
                    sys.stdout.flush()
                    dummytimestamp = int(row[0].split()[3])
                    if event_index > 0:
                        if dummytimestamp<timestamp[event_index-1]-nTimeResets*2147483647:
                            nTimeResets += 1
                    timestamp[event_index]=(dummytimestamp+nTimeResets*2147483647)
                if line_index%8 == 0:#every eigth row is the data
                    dummy = row[0].split()
                    dummy=[int(i) for i in dummy]
                    samples[event_index] = np.array(dummy ,dtype=np.int16)
                    samples[event_index] -= int(sum(samples[event_index][0:20])/20)
                    #check the polarity and check if the pulse crosses threshold
                    peak_index[event_index] = np.argmax(np.absolute(samples[event_index]))
                    if np.absolute(samples[event_index][peak_index[event_index]]) >= threshold:
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
                            sg = int((edges[event_index][1]-edges[event_index][0])*0.7)
                            lg = int((edges[event_index][1]-edges[event_index][0])*0.9)
                            #shortgate[event_index] = np.trapz(samples[event_index][refpoint[event_index]-5:refpoint[event_index]+sg])
                            shortgate[event_index] = np.trapz(samples[event_index][edges[event_index][0]:edges[event_index][0]+sg])
                            #longgate[event_index] = np.trapz(samples[event_index][refpoint[event_index]-5:refpoint[event_index]+lg])
                            longgate[event_index] = np.trapz(samples[event_index][edges[event_index][0]:edges[event_index][0]+lg])
                            event_index += 1
        #throw away empty rows.
        samples = samples[0:event_index]
        timestamp = timestamp[0:event_index]
        height = height[0:event_index]
        peak_index = peak_index[0:event_index]
        edges = edges[0:event_index]
        area = area[0:event_index]
        shortgate = shortgate[0:event_index]
        longgate = longgate[0:event_index]
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
                         'shortgate' : shortgate,
                         'longgate' : longgate,
                         'refpoint' : refpoint,
                         'left' : left,
                        'right' : right})

def cfd(samples, frac):
    peak = np.max(samples)
    refpoint = 0
    for i in range(0,len(samples)):
        if samples[i]>= frac*peak:
            refpoint = i
            break
    return refpoint

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

# def adv_framer(frame, cfd_frac=0.5, keep_samples = False):
#     nTimeResets = 0
#     timestamp = np.array([0]*len(frame), dtype=np.int64)
#     area  = np.array([0]*len(frame), dtype=np.int16)
#     short = np.array([0]*len(frame), dtype=np.int16)
#     refpoint = np.array([0]*len(frame), dtype=np.int16)
#     edges = [None]*len(frame)

#     for n in range(0,len(frame)):
#         k=100*n/(len(frame))+1
#         sys.stdout.write("\rGenerating more processed dataframe %d%%" % k)
#         sys.stdout.flush()
#         if n>0:
#             if frame.timestamp[n]<frame.timestamp[n-1]:
#                 nTimeResets += 1
#         timestamp[n]=(frame.timestamp[n]+nTimeResets*2147483647)
#         refpoint[n] = cfd(frame.samples[n], cfd_frac)
#         edges[n] = find_edges(frame.samples[n], refpoint[n])
#         area[n] = np.trapz(frame.samples[n][edges[n][0]:edges[n][1]])
#         #sg=int((edges[n][1]-edges[n][0])*0.95)
#         sg=7
#         if frame.samples[n][edges[n][0]]+sg<len(frame.samples[n]):
#             short[n] = np.trapz(frame.samples[n][edges[n][0]:edges[n][0]+sg])
#     if keep_samples == True:
#         return pd.DataFrame({'timestamp' : timestamp,
#                              'refpoint' : refpoint,
#                              'samples' : frame.samples,
#                              'height' : frame.height,
#                              'edges' : edges,
#                              'area' : area,
#                              'short' : short})
#     elif keep_samples == False:
#         return pd.DataFrame({'timestamp' : timestamp,
#                              'refpoint' : refpoint,
#                              'height' : frame.height,
#                              'edges' : edges,
#                              'area' : area,
#                              'short' : short})

def tof_spectrum(ne213, yap, fac=8, tolerance = 100):
    ymin=0
    tof_hist = np.histogram([], 2*tolerance, range=(-tolerance, tolerance))
    tof_hist1 = np.histogram([], 2*tolerance, range=(-tolerance, tolerance))
    tof_hist2 = np.histogram([], 2*tolerance, range=(-tolerance, tolerance))

    #for ne in range(0, len(ne213)):
    counter=0
    for row in ne213.itertuples():
        ne=row[0]
        counter += 1
        k = 100*counter/len(ne213)
        sys.stdout.write("\rGenerating tof spectrum %d%%" % k)
        sys.stdout.flush()
        for y in range(ymin, len(yap)):
            Delta=(fac*ne213.timestamp[ne]+ne213.refpoint[ne])-(fac*yap.timestamp[y]+yap.refpoint[y])
            if Delta > tolerance:
                ymin = y
            if -tolerance < Delta <tolerance:
                tof_hist[0][tolerance+int(Delta)] += 1
            elif Delta < -tolerance:
                break
    return tof_hist


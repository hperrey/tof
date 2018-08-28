import numpy as np
import pandas as pd
import csv
import sys


def basic_framer(filename, threshold):
    #Get number of lines
    nlines=sum(1 for line in (open(filename)))

    nevents = int(nlines/8)
    samples = [None]*nevents
    timestamp = np.array([0]*nevents, dtype=np.int64)

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
                    timestamp[event_index] = int(row[0].split()[3])
                if line_index%8 == 0:#every eigth row is the data
                    dummy = row[0].split()
                    dummy=[int(i) for i in dummy]
                    samples[event_index] = np.array(dummy ,dtype=np.int16)
                    samples[event_index] -= int(sum(samples[event_index][0:20])/20)
                    #check the polarity and check if the pulse crosses threshold
                    peak_index = np.argmax(np.absolute(samples[event_index]))
                    if np.absolute(samples[event_index][peak_index]) >= threshold:
                        if samples[event_index][peak_index] < 0:
                            samples[event_index] *= -1
                        event_index += 1
        samples=samples[0:event_index]
        timestamp=timestamp[0:event_index]
    except IOError:
        return None
    return pd.DataFrame({'timestamp': timestamp, 'samples' : samples})

def cfd(samples, frac):
    peak = np.max(samples)
    refpoint = 0
    for i in range(0,len(samples)):
        if samples[i]>= frac*peak:
            refpoint = i
            break
    return refpoint

def adv_framer(frame, cfd_frac=0.5):
    nTimeResets = 0
    timestamp = np.array([0]*len(frame), dtype=np.int64)
    refpoint = np.array([0]*len(frame), dtype=np.int16)
    for n in range(0,len(frame)):
        k=100*n/(len(frame))+1
        sys.stdout.write("\rGenerating more processed dataframe %d%%" % k)
        sys.stdout.flush()
        if n>0:
            if frame.timestamp[n]<frame.timestamp[n-1]:
                nTimeResets += 1
        timestamp[n]=(frame.timestamp[n]+nTimeResets*2147483647)
        refpoint[n] = cfd(frame.samples[n], cfd_frac)
    return pd.DataFrame({'timestamp': timestamp,
                        'refpoint': refpoint})

def tof_spectrum(ne213, yap, fac=16, tolerance = 100):
    ymin=0
    tof_hist = np.histogram([], 2*tolerance, range=(-tolerance, tolerance))
    for ne in range(0, len(ne213)):
        k = 100*ne/len(ne213)
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

import numpy as np
import pandas as pd
import csv
import sys
from itertools import islice
import time
#import matplotlib.pyplot as plt
from math import sqrt
from math import atan

def load_data(filename, threshold, frac=0.3, nlines=0, startline=0, nTimesReset=0, no_skip=False, chunksize=2**18, outpath='data/chunk'):
    t0 = time.time() 
    print("Scanning the file to get number of chunks:")
    nChunks = int(round(0.5 + sum(1 for row in open(filename, 'r'))/chunksize))
    t1 = time.time()
    print("Scan time: ", t1-t0, ' seconds')
    print("Will generate ", nChunks)
    Chunks = pd.read_csv(filename, header=None, usecols=[5,7], names=['timestamp', 'samples'], chunksize=chunksize)
    count=0
    tdummy1=t1
    for df in Chunks:
        tdummy2=tdummy1
        print("Chunk number", count + 1, "/", nChunks)
        df['samples'] = df.samples.str.split().apply(lambda x: np.array(x, dtype=np.int16))

        samples = np.array([None]*df.shape[0])
        timestamp = np.array([0]*df.shape[0], dtype=np.int64)
        amplitude = np.array([0]*df.shape[0], dtype=np.int16)
        peak_index = np.array([0]*df.shape[0], dtype=np.int16)
        valid_event = np.array([True]*df.shape[0], dtype=np.int16)
        ref_point_rise = np.array([0]*df.shape[0], dtype=np.int32)
        ref_point_fall = np.array([0]*df.shape[0], dtype=np.int32)
        nTimesReset = 0

        for i in range(0, df.shape[0]):
            u = chunksize*count + i
            k = round(100*i/df.shape[0])
            sys.stdout.write("\rGenerating dataframe %d%%" % k)
            sys.stdout.flush()

            Baseline = int(round(np.average(df.samples[u][0:20])))
            peak_index[i] = np.argmin(df.samples[u])

            #Check that only events above threshold are accepted and that first 20 samples can give a good baseline.
            if abs(df.samples[u][peak_index[i]] - Baseline) < threshold or (max(df.samples[u][0:20])-min(df.samples[u][0:20])) > 3:
                valid_event[i] = False
                continue
            else:
                samples[i] = df['samples'][u] - Baseline
                ref_point_rise[i], ref_point_fall[i] = cfd(samples=samples[i], frac=frac, peak_index=peak_index[i])
                amplitude[i] = samples[i][peak_index[i]]
                timestamp[i] = df['timestamp'][u]
                if i > 0:
                    if timestamp[i] < timestamp[i-1]-nTimesReset*2147483647:
                        nTimesReset += 1
                    timestamp[i] += nTimesReset*2147483647
        df['timestamp'] = timestamp
        df['samples'] = samples
        df['valid_event'] = valid_event
        df['amplitude'] = amplitude
        df['peak_index'] = peak_index
        df['ref_point_rise'] = ref_point_rise
        df['ref_point_fall'] = ref_point_fall
        df = df.query('valid_event == True').reset_index()
        df.to_hdf(outpath+'.h5', key='key%s'%count)
        df = df.drop('samples', axis = 1)
        df.to_hdf(outpath+'cooked.h5', key='key%s'%count)
        tdummy1=time.time()
        print('chunk ', count, ' processed in ', tdummy1-tdummy2, ' seconds'  )
        count += 1




def basic_framer(filename, threshold, frac=0.3, nlines=0, startline=0, nTimesReset=0, no_skip=False):
    #Get number of lines
    if nlines == 0:
        nlines=sum(1 for line in (open(filename)))
    nevents = int(nlines/8)
    samples = [None]*nevents
    timestamp = np.array([0]*nevents, dtype=np.int64)
    refpoint_rise = np.array([0]*nevents, dtype=np.int32)
    refpoint_fall = np.array([0]*nevents, dtype=np.int32)
    peak_index = np.array([0]*nevents, dtype=np.int16)
    height = np.array([0]*nevents, dtype=np.int16)
    acq_window = 0
    try:
        with open(filename, newline='\n') as f:
            reader = csv.reader(f)
            line_index = startline
            event_index = 0
            idx=0
            #Scan forwards through the file until you are at the provided start time
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
                        if dummytimestamp < timestamp[event_index-1]-nTimesReset*2147483647:
                            nTimesReset += 1
                            #print('\ntime reset!\n')
                    timestamp[event_index]= (dummytimestamp+nTimesReset*2147483647)
                if line_index%8 == 0:#every eigth row is the data
                    dummy = row[0].split()
                    dummy=[int(i) for i in dummy]
                    samples[event_index] = np.array(dummy ,dtype=np.int16)
                    #B is the number of samples used to calculate baseline
                    #We don't care about events that have large peaks or noise in this interval
                    B = 20
                    baseline = int(sum(samples[event_index][0:B])/B)
                    samples[event_index] -= baseline
                    #check the polarity and check if the pulse crosses threshold and if it is properly contained
                    peak_index[event_index] = np.argmax(np.absolute(samples[event_index]))
                    if (np.absolute(samples[event_index][peak_index[event_index]]) < threshold and no_skip==False):
                        continue
                    else:
                        if samples[event_index][peak_index[event_index]] < 0:
                            samples[event_index] *= -1
                        #get pulse height and pulse edge bins
                        height[event_index] = samples[event_index][peak_index[event_index]]
                        refpoint_rise[event_index], refpoint_fall[event_index] = cfd(samples[event_index], frac, peak_index[event_index])
                        #throw away events marked problematic by cfd alg. and events without room for tail.
                        if ((refpoint_rise[event_index]<0 and no_skip==False) or  (refpoint_fall[event_index]<0 and no_skip==False)):
                            continue
                        event_index += 1
        #throw away empty rows.
        samples = samples[0:event_index]
        timestamp = timestamp[0:event_index]
        height = height[0:event_index]
        peak_index = peak_index[0:event_index]
        refpoint_rise = refpoint_rise[0:event_index]
        refpoint_fall = refpoint_fall[0:event_index]
    except IOError:
        return None
    return pd.DataFrame({'timestamp': timestamp,
                         'samples' : samples,
                         'height' : height,
                         'peak_index':peak_index,
                         'refpoint_rise' : refpoint_rise,
                         'refpoint_fall' : refpoint_fall}), nTimesReset

def get_gates(frame, lg=500, sg=55, offset=10):
    longgate=np.array([0]*len(frame), dtype=np.int16)
    shortgate=np.array([0]*len(frame), dtype=np.int16)
    pulsetail=np.array([0]*len(frame), dtype=np.int16)
    theta=np.array([0]*len(frame), dtype=np.int16)
#    species=np.array([-1]*len(frame), dtype=np.int8)
    for i in range(0, len(frame)):
        k = round(100*i/len(frame))
        sys.stdout.write("\rCalculating gates %d%%" % k)
        sys.stdout.flush()

        #start = int(round(frame.refpoint_rise[i]/1000))-offset
        start = frame.peak_index[i]-offset
        longgate[i] = np.trapz(frame.samples[i][start:start+lg])
        shortgate[i] = np.trapz(frame.samples[i][start:start+sg])
        theta[i] = atan(2*frame.height[i]/frame.refpoint_fall[i]-frame.refpoint_rise[i])

        #send weird events to quarantine bins
        if shortgate[i]>longgate[i]:
            #workaround. need to deal with reflections properly!
            longgate[i]=20000
            shortgate[i]=20000
        if longgate[i]<=0 or shortgate[i]<=0:
            longgate[i]=20000
            shortgate[i]=20000

        #tail
        pulsetail[i] = np.trapz(frame.samples[i][int(frame.refpoint_fall[i]/1000):int(frame.refpoint_fall[i]/1000)+lg])

    frame['ps'] = (longgate-shortgate)/longgate
    frame['longgate']=longgate
    frame['shortgate']=shortgate
    frame['pulsetail']=pulsetail
    frame['theta']=theta
    return 0



def get_species(df, X=[0, 1190,2737, 20000], Y=[0, 0.105, 0.148, 0.235]):
    species=np.array([-1]*len(df), dtype=np.int8)
    #loop through pulses
    for n in range(0, len(df)):
        k = round(100*n/len(df))
        sys.stdout.write("\rGetting species %d%%" % k)
        sys.stdout.flush()
        #If we are to the left of the exclusion zone
        if df.longgate[n]<X[1]:
            #inside exclusion box=>indistinguishable
            if df.ps[n]<Y[1]:
                species[n]=-1
                #above exclusion box=>neutron
            else:
                species[n]=1
        #If we are to the right of the exclusion zone
        #then loop through coordinates
        elif df.longgate[n]>=X[1]:
            for i in range(1,len(X)):
                #find the interval the pulse belongs to
                if df.longgate[n]<X[i]:
                    if X[i]>=X[1]:
                        #are we below(gamma) or above(neutron) of the discrimination line
                        if df.ps[n]<Y[i-1]+(df.longgate[n]-X[i-1])*(Y[i]-Y[i-1])/(X[i]-X[i-1]):
                            species[n] = 0
                        else:
                            species[n] = 1
                        break
    df['species'] = species

def cfd(samples, frac, peak_index):
    peak = samples[peak_index]
    print('frac*peak = %d0'%(peak*frac))
    rise_index = 0
    fall_index = 0
    #find the cfd rise point
    for i in range(0, peak_index):
        if samples[i] < frac * peak:
            rise_index = i
            break
        else:
            rise_index = 0
        #find the cfd fall point
        for i in range(peak_index, len(samples)):
            if samples[i] > frac*peak:
                fall_index = i
                break
            else:
                fall_index = 0
        slope_rise = (samples[rise_index] - samples[rise_index-1])#divided by 1ns
        slope_fall = (samples[fall_index] - samples[fall_index-1])#divided by 1ns
        #slope equal 0 is a sign of error. fx a pulse located
        #in first few bins and already above threshold in bin 0.
        #rise
        if slope_rise == 0:
            print('\nslope == 0!!!!\nindex=', rise_index,'\n', samples[rise_index-5:rise_index+5])
            tfine_rise = -1
        else:
            tfine_rise = 1000*(rise_index-1) + int(round(1000*(peak*frac-samples[rise_index-1])/slope_rise))
            #fall
        if slope_fall == 0:
            print('\nslope == 0!!!!\nindex=', fall_index,'\n', samples[fall_index-5:fall_index+5])
            tfine_fall = -1
        else:
            tfine_fall = 1000*(fall_index-1) + int(round(1000*(peak*frac-samples[fall_index-1])/slope_fall))
        return tfine_rise, tfine_fall


def cfd(samples, frac, peak_index):
    peak = samples[peak_index]
    rise_index = 0
    fall_index = 0
    #find the cfd rise point
    for i in range(0, peak_index):
        if samples[i] < frac * peak:
            rise_index = i
            break
        else:
            rise_index = 0
    #find the cfd fall point
    for i in range(peak_index, len(samples)):
        if samples[i] > frac*peak:
            fall_index = i
            break
        else:
            fall_index = 0
    slope_rise = (samples[rise_index] - samples[rise_index-1])#divided by 1ns
    slope_fall = (samples[fall_index] - samples[fall_index-1])#divided by 1ns
    #slope equal 0 is a sign of error. fx a pulse located
    #in first few bins and already above threshold in bin 0.
    #rise
    if slope_rise == 0:
        print('\nslope == 0!!!!\nindex=', rise_index,'\n', samples[rise_index-5:rise_index+5])
        tfine_rise = -1
    else:
        tfine_rise = 1000*(rise_index-1) + int(round(1000*(peak*frac-samples[rise_index-1])/slope_rise))
    #fall
    if slope_fall == 0:
        print('\nslope == 0!!!!\nindex=', fall_index,'\n', samples[fall_index-5:fall_index+5])
        tfine_fall = -1
    else:
        tfine_fall = 1000*(fall_index-1) + int(round(1000*(peak*frac-samples[fall_index-1])/slope_fall))
    return tfine_rise, tfine_fall


def get_frames(filename, threshold, frac=0.3, no_skip=False, outpath='/home/rasmus/Documents/ThesisWork/code/tof/data/'):
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
    #FrameList=[0]*len(Blocklines)
    nTimesReset = 0
    for i in range(0, (nBlocks+1)):
        print('\n -------------------- \n frame', i+1, '/', (nBlocks+1), '\n --------------------')
        Frame, nTimesReset = basic_framer(filename, threshold, frac, nlines=Blocklines[i], startline=i*nlinesBlock, nTimesReset=nTimesReset, no_skip=no_skip)
        get_gates(Frame)
        get_species(Frame)
        if outpath!='':
            Frame.to_hdf(outpath+'%s.h5'%i, 'a')
    time1 = time.time()
    deltaT=time1-time0

    #make the cooked frame
    D=[0]*(nBlocks+1)
    for i in range(0, (nBlocks+1)):
        D[i]=pd.read_hdf(outpath+'%d.h5'%i)
        D[i].drop('samples', axis=1)
    D=pd.concat(D).reset_index()
    D.to_hdf(outpath+'_cooked.h5', 'a')

    print('Runtime: ', deltaT/60, 'minutes')
    return 0



def tof_spectrum(ne213, yap, fac=8, tol_left=0, tol_right=120):
    ymin=0
    tof_hist = np.histogram([], tol_left+tol_right, range=(tol_left, tol_right))
    dt=np.array([0]*len(ne213), dtype=np.int32)
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
            Delta=int(round(((fac*1000*ne213.timestamp[ne]+ne213.refpoint_rise[ne])-(fac*1000*yap.timestamp[y]+yap.refpoint_rise[y]))/1000))
            if Delta > tol_right:
                ymin = y
            if tol_left <= Delta < tol_right:
                tof_hist[0][tol_left+int(Delta)] += 1
                if dt[ne] == 0:
                    dt[ne]=Delta
                else:
                    print('Multiple matches!!! taking the first one!')
            elif Delta < -tol_right:
                break
        ne213['dt']=dt
    return tof_hist


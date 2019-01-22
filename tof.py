import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import csv
import sys
import os


def dask_chewer(filename, outpath, threshold, maxamp, lg=500, sg=60, blocksize=25*10**6, mode=0):
    """Uses multprocessing to process data and return a simple dataframe in parquet format"""
    filesize = os.path.getsize(filename)
    nBlocks = int(round(0.5 + (filesize / blocksize) ) )
    print('Filesize = ', filesize, 'Bytes ...', nBlocks, 'Blocks will be generated' )
    #Load the csv file
    df = dd.read_csv(filename, header=None, usecols=[3, 5, 7], names=['event_number','timestamp', 'samples'],
                    dtype={'idx': np.int64, 'timestamp': np.int64, 'samples': np.object}, blocksize=blocksize)
    #format samples
    df['samples'] = df['samples'].str.split().apply(lambda x: np.array(x, dtype=np.int16), meta=df['samples'])
    df['samples'] = df['samples'].apply(lambda x: x - int(round(np.average(x[0:20]))), meta=df['samples'])

    #Throw away events whose amplitude is below the threshold and those whose amplitude
    #is greater or equal to the expected maximum amplitude (likely have their tops cut off)
    df['amplitude'] = df['samples'].apply(lambda x: np.max(np.absolute(x)), meta=df['samples']).astype(np.int16)
    df = df[df['amplitude'] > threshold]
    df = df[maxamp > df['amplitude']]
    df['peak_index'] = df['samples'].apply(lambda x: np.argmin(x), meta=df['samples']).astype(np.int16)

    #charge integrals
    if mode == 0:
        #integrate over the absolute value of the pulses
        df['qdc_lg'] = df['samples'].apply(lambda x: np.trapz(np.abs(x[np.argmin(x)-10:np.argmin(x)+lg])), meta=df['samples']).astype(np.int16)
        df['qdc_sg'] = df['samples'].apply(lambda x: np.trapz(np.abs(x[np.argmin(x)-10:np.argmin(x)+sg])), meta=df['samples']).astype(np.int16)
    elif mode == 1:
        # integrate the pulses as they are (negative bins will drag the integral down)
        df['qdc_lg'] = df['samples'].apply(lambda x: abs(np.trapz(x[np.argmin(x)-10:np.argmin(x)+lg])), meta=df['samples']).astype(np.int16)
        df['qdc_sg'] = df['samples'].apply(lambda x: abs(np.trapz(x[np.argmin(x)-10:np.argmin(x)+sg])), meta=df['samples']).astype(np.int16)
    elif mode == 2:
        #integrate the pulses, but ignore negative bins
        #Not working properly yet
        df['qdc_lg'] = df['samples'].apply(lambda x: abs(np.trapz(x[np.argmin(x)-10:np.argmin(x)+lg].clip(max = 0)), meta=df['samples'])).astype(np.int16)
        df['qdc_sg'] = df['samples'].apply(lambda x: abs(np.trapz(x[np.argmin(x)-10:np.argmin(x)+sg].clip(max = 0)), meta=df['samples'])).astype(np.int16)

    #cfd triggers
    #...

    #save to disk
    with ProgressBar():
        print('Dataframe generated: Saving dataframe to disk')
        df.to_parquet(outpath, engine='pyarrow')
    return df

def process_daskframe(filename, frac=0.3, lg=500, sg=55, offset=10, mode=2):
    """Reads the parquet file(s) produced by dask_chewer and adds longgate and shortgate integrals as well as cfd rise and fall triggers"""
    t0 = time.time()
    #read the file(s)
    df = pd.read_parquet(filename, engine='pyarrow')
    #Get df length
    L = len(df)

    #define the arrays used for the new columns, as well as some dummy variables
    rise = np.array([0]*L, dtype=np.int32)
    fall = np.array([0]*L, dtype=np.int32)
    longgate = np.array([0]*L, dtype=np.int32)
    shortgate = np.array([0]*L, dtype=np.int32)
    ps = np.array([0]*L, dtype=np.float32)
    #timestamp = np.array([0]*L, dtype=np.int64)
    #timestampDummy= 0
    #nTimesReset = 0

    #loop through df and populate arrays
    for i in range(0, L):
        if i%1000 == 0:
            k = int(100*i/L + 0.5)
            sys.stdout.write("\rGenerating qdc integrals and cfd triggers %d%%" % k)
            sys.stdout.flush()
        rise[i], fall[i] = cfd(df.samples[i], frac, df.peak_index[i])
        longgate[i], shortgate[i], ps[i] = pulse_integrate(df, i=i, lg=lg, sg=sg, offset=offset, mode=mode)
        #Correct the timestamp resetting done by Wavedump at t=2**32
        #if (df['timestamp'][i] < timestampDummy):
        #    nTimesReset += 1
        #timestamp[i] = df['timestamp'][i] + nTimesReset*2147483647
        #timestampDummy = df['timestamp'][i]
    sys.stdout.write("\rGenerating qdc integrals and cfd triggers 100%%")
    sys.stdout.flush()

    #add the new columns and throw away poorly contained events
    df['rise'], df['fall'] = rise, fall
    df['longgate'], df['shortgate'], df['ps'] = longgate, shortgate, ps
    df['timestamp'] = timestamp
    #Let the user know the runtime
    print('\nprocessing time: ',time.time() - t0)

    return df


def load_data(filename, threshold, frac=0.3, skip_badevents=True, chunksize=2**16, outpath='data/chunk'):
    """load_data()\nArguments and default inputs: \nfilename: path to datafile, \nthreshold: absolute value, unit ADC count, range 0 to 1023, \nskip_badevents=True: Wether to skip events where baseline was noisy or threshold was not surpassed, \nchunksize=2**16: the size of the chunks. for 8gB RAM 2**16-2**17 seems to be the limit, \noutpath='data/chunk': path to outputfile location.:"""
    t0 = time.time()
    print("Scanning the file to get number of chunks:")
    nChunks = int(round(0.5 + sum(1 for row in open(filename, 'r'))/chunksize))
    t1 = time.time()
    print("Scan time: ", t1-t0, ' seconds')
    print("Will generate ", nChunks)
    Chunks = pd.read_csv(filename, header=None, usecols=[3, 5, 7], names=['idx', 'timestamp', 'samples'], chunksize=chunksize)
    count=0
    tdummy1=t1
    nTimesReset = 0
    
    for df in Chunks:
        df= df.reset_index()
        tdummy2=tdummy1
        print("Chunk number", count + 1, "/", nChunks)
        df['samples'] = df.samples.str.split().apply(lambda x: np.array(x, dtype=np.int16))

        #dummy variable used to compare consecutive timestamps. reset to zero when processing a new chunk of the df
        timestampDummy = 0
        #Arrays for the data we will put into the df chunks columns.
        samples = np.array([None]*df.shape[0])
        timestamp = np.array([0]*df.shape[0], dtype=np.int64)
        amplitude = np.array([0]*df.shape[0], dtype=np.int16)
        peak_index = np.array([0]*df.shape[0], dtype=np.int16)
        valid_event = np.array([True]*df.shape[0], dtype=np.int16)
        ref_point_rise = np.array([0]*df.shape[0], dtype=np.int32)
        ref_point_fall = np.array([0]*df.shape[0], dtype=np.int32)

        for i in range(0, df.shape[0]):
            #u = chunksize*count + i
            k = round(100*i/df.shape[0])
            sys.stdout.write("\rGenerating dataframe %d%%" % k)
            sys.stdout.flush()

            Baseline = int(round(np.average(df.samples[i][0:20])))
            peak_index[i] = np.argmin(df.samples[i])

            #Accept only only events above threshold and for which the first 20 samples can give a good baseline.
            if (skip_badevents==True) and (abs(df.samples[i][peak_index[i]] - Baseline) < threshold or (max(df.samples[i][0:20])-min(df.samples[i][0:20])) > 3):
                valid_event[i] = False
                continue
            else:
                #subtract baseline, get cfd refpoint and get pulse amplitude.
                samples[i] = df['samples'][i] - Baseline
                ref_point_rise[i], ref_point_fall[i] = cfd(samples=samples[i], frac=frac, peak_index=peak_index[i])
                amplitude[i] = samples[i][peak_index[i]]
                #Correct the timestamp resetting done by Wavedump at t=2**32
                if ((df['timestamp'][i] + nTimesReset*2147483647) < timestampDummy):
                    nTimesReset += 1
                timestamp[i] = df['timestamp'][i] + nTimesReset*2147483647
                timestampDummy = timestamp[i]

        #Here we add the new columns to the chunk
        df['timestamp'] = timestamp
        df['samples'] = samples
        df['valid_event'] = valid_event
        df['amplitude'] = amplitude
        df['peak_index'] = peak_index
        df['ref_point_rise'] = ref_point_rise
        df['ref_point_fall'] = ref_point_fall
        #We Only keep events that are valid
        df = df.query('valid_event == True').reset_index()
        df.to_hdf(outpath+'.h5', key='key%s'%count)
        df = df.drop('samples', axis = 1)
        df.to_hdf(outpath+'cooked.h5', key='key%s'%count)
        tdummy1=time.time()
        print('chunk ', count+1, ' processed in ', tdummy1-tdummy2, ' seconds'  )
        count += 1
    tf = time.time()
    print("total processing time: ", tf-t0)



def pulse_integrate(frame, i, lg=500, sg=55, offset=10, mode=0):
    """Integrates pulses over two timescales: lg and sg given in nanoseconds, starting offset ns from the pulse peak.
    Integration is done in one of three ways.\n mode=0: integrate across positive and negative bins alike.
    \n mode=1 integrate positive bins only. \n mode=2 integrate the absolute value of the pulse."""
    longgate = 0
    shortgate = 0
    #start = int(round(frame.refpoint_rise[i]/1000))-offset
    start = frame.peak_index[i]-offset
    if mode == 0:
        longgate = abs(np.trapz(frame.samples[i][start:start+lg]))
        shortgate = abs(np.trapz(frame.samples[i][start:start+sg]))
    elif mode == 1:
        longgate = abs( np.trapz( frame.samples[i][start : start + lg].clip(max = 0) ) )
        shortgate = abs( np.trapz( frame.samples[i][start : start + sg].clip(max = 0) ) )
    elif mode == 2:
        longgate = abs( np.trapz( np.absolute( frame.samples[i][start : start + lg] ) ) )
        shortgate = abs( np.trapz( np.absolute( frame.samples[i][start : start + sg] ) ) )

    #send weird events to quarantine bins
    if shortgate > longgate:
        longgate = 40000
        shortgate = 40000
    if longgate <= 0 or shortgate <= 0:
        longgate = 40000
        shortgate = 40000

    return longgate, shortgate, (longgate-shortgate)/longgate



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






def tof_spectrum(ne213, yap, fac=8, tol_left=0, tol_right=120):
    tol_left = tol_left * 1000
    tol_right = tol_right * 1000
    ymin=0
    tof_hist = np.histogram([], tol_left+tol_right, range=(tol_left, tol_right))
    dt=np.array([0]*len(ne213), dtype=np.int32)

    counter=0
    #loop through all ne213 events
    for row in ne213.itertuples():
        ne=row[0]
        counter += 1
        k = 100*counter/len(ne213)
        sys.stdout.write("\rGenerating tof spectrum %d%%" % k)
        sys.stdout.flush()
        #loop from min yap index until break condition applies
        for y in range(ymin, len(yap)):
            Delta = (fac*1000*ne213.timestamp[ne]+ne213.ref_point_rise[ne]) - (fac*1000*yap.timestamp[y]+yap.ref_point_rise[y])
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


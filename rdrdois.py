import numpy as np
import pandas as pd
import logging 
import csv
import sys

def baseliner2(Samples):
    B=0
    baseline_sample_range=20
    B=sum(Samples[0:baseline_sample_range])/baseline_sample_range
    return B

def load_events(filename,treshold):
    "Load Events written by WaveDump and save them to a dataframe. Stores the timestamp and all the samples for each event."
    log = logging.getLogger('load_events')
    log.setLevel(logging.WARNING)

    Nlines=sum(1 for line in (open(filename)))


    log.info("Reading data from file '" + filename + "'")
    samples = [None]*int(Nlines/8)#we have seven header lines and one data line for each event
    time = [None]*int(Nlines/8)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #treshold = 25
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    try:
        with open(filename, newline='\n') as f:
            reader = csv.reader(f)
            n=0#row number
            N=0#event number
            #dt=1#nanosecond
            B=[None]*int(Nlines/8)
            for row in reader:
                k=100*n/Nlines+1
                sys.stdout.write("\rGenerating basic dataframe %d%%" % k)
                sys.stdout.flush()
                #if n%10000==0:
                #    print('row number ', n,'/',Nlines)
                n+=1#start counting rows from 1
                #Extract timestamp
                if n%8==6: #every sixth row is the timestamp. Each row is a list containing a single string element.
                    log.debug('parsing the Timestamp')
                    time[N] = int(row[0].split()[3])#we split the string and convert it to an int
                #Extract the samples
                if n%8==0:#every eigth row is the data
                    log.debug('loading the data')
                    dummy = row[0].split()
                    dummy=[int(i) for i in dummy]
                    samples[N]=np.array(dummy ,dtype='float64')
                    B[N]=baseliner2(samples[N])
                    samples[N]-=B[N]
                    #check the polarity and check if the pulse crosses treshold
                    peak_index = np.argmax(abs(samples[N]))
                    if samples[N][peak_index] < 0:
                        samples[N]*=-1
                    if samples[N][peak_index]>=treshold:
                        N+=1
                        #and otherwise samples[N] is overwritten on next iteration
            max_index=0
            for i in range(0, int(Nlines/8)):
                if time[i]==None:
                    max_index=i-1
                    break
            samples=samples[0:max_index]
            time=time[0:max_index]
            B=B[0:max_index]
        #save to dataframe
        log.debug('done reading data, saving to dataframe')            
        Frame=pd.DataFrame({'TimeStamp': time, 'Samples' : samples, 'Baseline':B})
        #print(len(Frame))
        log.debug('dataframe created')
    except IOError:
        log.error('failed to read file')
        return None
    return Frame

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 09:49:37 2018

@author: rasmus
"""
import logging 
import csv
import numpy as np
import pandas as pd

#seems worse
def baseliner1(Samples):
    B=0
    H =np.histogram(Samples, int(max(Samples)))
    for i in range(0,5):
        B+=i*H[0][i]
    B/=sum(H[0][0:5])
    return B

#seems better
def baseliner2(Samples):
    B=0
    baseline_sample_range=20
    B=sum(Samples[0:baseline_sample_range])/baseline_sample_range
    return B

logging.basicConfig()
def load_events(filename):
    "Load Events written by WaveDump and save them to a dataframe. Stores the timestamp and all the samples for each event."
    log = logging.getLogger('load_events')
    log.setLevel(logging.WARNING)
    Nlines=sum(1 for line in (open(filename)))
    log.info("Reading data from file '" + filename + "'")
    samples = [0]*int(Nlines/8)#we have seven header lines and one data line for each event
    time = [0]*int(Nlines/8)
    try:
        with open(filename, newline='\n') as f:
            reader = csv.reader(f)
            n=0#row number
            N=0#event number
            #dt=1#nanosecond
            B=[0]*int(Nlines/8)
            for row in reader:
                if n%10000==0:
                    print('row number ', n,'/',Nlines)
                N=int(n/8)#start counting events from zero  
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
                    peak_index = np.argmax(abs(samples[N]))
                    #check the polarity
                    if samples[N][peak_index] < 0:
                        samples[N]*=-1

        #save to dataframe
        log.debug('done reading data, saving to dataframe')            
        Frame=pd.DataFrame({'TimeStamp': time, 'Samples' : samples, 'Baseline':B})
        log.debug('dataframe created')
    except IOError:
        log.error('failed to read file')
        return None
    return Frame


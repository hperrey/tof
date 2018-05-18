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
            dt=1#nanosecond
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
                    dummy=[int(i) for i in dummy]#should I specify datatype here.
                    samples[N]=np.array(dummy ,dtype='float64')#include ,dtype='int16' for five times smaller dataframe, but five times slower program.
              
        #save to dataframe         
        log.debug('done reading data, saving to dataframe')            
        Frame=pd.DataFrame({'TimeStamp': time, 'Samples' : samples})
        log.debug('dataframe created')
    except IOError:
        log.error('failed to read file')
        return None
    return Frame


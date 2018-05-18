import logging 
import numpy as np
import pandas as pd



def shifter(wave0):
    "Takes a dataframe of the form produced by Reader.py and processe it. Finds the zerocrossing and the end of the pulse. Stores Timestamp,zero crossing to the left and right, and all samples in between. Has the Following columns: TimeStamp, Samples, Tleft, TrightExample of how to use: A=shifter(some_frame); plt.plot(A.Samples[15]);#to plot the 15 event from left to right zerocrossing."
    log = logging.getLogger('shifter')
    log.setLevel(logging.WARNING)
    SampleLength=len(wave0.Samples[0])
    NumSamples=len(wave0)
    samples = [0]*NumSamples
    Tleft=[None]*NumSamples#zero crossing rise
    Tright=[None]*NumSamples#zero crossing fall
    Area=[0]*NumSamples
    for n in range(0,NumSamples):
        log.info("Reading dataframe")
        if n%1000==0:
            print('processing Event ', n, ' out of ', NumSamples, ' events')
        flag=0
        rise=True
        for i in range(0,SampleLength):
            #find left zero crossing Tleft
            if rise==True:
                if flag>5:
                    continue
                elif flag == 5:    #-4 brings us to the first flag. Take a
                    Tleft[n] =i-6#few more steps back to reach zero crossing.
                    rise=False
                elif flag<5 and wave0.Samples[n][i]>10:
                    flag+=1
                else:
                    flag=0
            #find right zero crossing Tright
            if rise==False and wave0.Samples[n][i]<5:
                Tright[n]=i+1
                break
 #       print(np.shape(samples))
#        print(np.shape(Area[1]))
        samples[n]=np.array(wave0.Samples[n][Tleft[n]:Tright[n]])
        Area[n]=0.5*(samples[n][0]+samples[n][-1])+sum(samples[n][1:-2])
    log.debug('done reading data, saving to dataframe')            
    wave0shifted=pd.DataFrame({'TimeStamp' : wave0.TimeStamp, 'Tleft': Tleft, 'Tright' : Tright,'Samples' : samples, 'Area': Area})
    log.debug('dataframe created')
    return wave0shifted

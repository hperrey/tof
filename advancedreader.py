import time
import pandas as pd
import cfd

def processframe(frame):
    #Start the timer
    tstart=time.time()
    #Initiate the time reset counter.
    nTimeResets=0
    #Initiate the columns of the advanced frame
    #(those which are not directly inherited from the simple frame)
    tstamp=[0]*len(frame)
    refpoint=[0]*len(frame)
    noevent=[False]*len(frame)
    lzcross=[0]*len(frame)
    rzcross=[0]*len(frame)

    #Loop through all the events
    for n in range(0,len(frame)):
        if n%1000==0:
            print('Event', n, '/', len(frame))
        #For every event after the first check if the time
        #has been reset. And keep count of the resets.
        if n>0:
            if frame.TimeStamp[n]<frame.TimeStamp[n-1]:
                nTimeResets+=1
        #Callibration to nanoseconds needed!
        #timestamp meas in clock cycles. clock freq = 0.125GHz
        #from clock cycle to ns. example: 80 ticks/8ticks/ns=10ns
        tstamp[n]=(frame.TimeStamp[n]+nTimeResets*2147483647)/8

        #Check if the event passes the threshold. This is
        #to sort out empty and also tiny events.
        if max(frame.Samples[n])<10:
            noevent[n]=True
            continue
        #For the remaining events (noevent=False) we
        #find a reference point, using the cfd script.
        else:
            refpoint[n] = cfd.shifter(frame.Samples[n])
            #And then we walk left and right to the zerocrossings.
            for u in range(refpoint[n]-1,-1,-1):
                if frame.Samples[n][u]<1:
                    lzcross[n]=u
                    break
            for y in range(refpoint[n]+1,len(frame.Samples[n])):
                if frame.Samples[n][y]<1:
                    rzcross[n]=y
                    break
  
    #Create the expanded dataframe.
    Frame=pd.DataFrame({'Timestamp': tstamp,
                        #'Samples' : frame.Samples,
                        #'Baseline' : frame.Baseline,
                        'Refpoint':refpoint,
                        #'Left':lzcross,
                        #'Right': rzcross,
                        'Noevent': noevent})

    #stop timing and print the runtime
    tstop=time.time()
    print('processing time = ',tstop-tstart)
    return Frame


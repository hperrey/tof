import numpy as np

#A = rdr.load_events('wave0.txt')
def inv(Samples):
    #B=bl.baseliner2(Samples)
    #Samples=Samples-B
    shift=4
    frac=0.40
    invsig=np.zeros(len(Samples))
    invsig[shift:]=frac*Samples[0:-shift]
    #invsig[0:-shift]=-frac*Samples[shift:]
    #invLead=np.zeros(len(Samples))
    #invLead[shift:]=-frac*Samples[0:-shift]
    return invsig-Samples#Samples+invLag, Samples+invLead



def crosser1(invsig):
    crossing = np.argmin(invsig)
    #cross_right = cross_left+np.argmin(inv2[cross_left:-1])
    #if cross_right<cross_left:
    #    print('!!!!!!!!!!!!!!!!!!!!!!!!!')
    return crossing#_left, cross_right


def shifter(Samples):
    peak=max(Samples)
    for i in range(0,len(Samples)):
        frac=0.5
        #frac=1
        if Samples[i]>=frac*peak:
            crossing=i
            break
    return crossing

#Super bad!!!
def crosser2(invsig, Samples):
    peakBin=np.argmax(Samples)
    zList=[]
    crossing=1029
    for i in range(1,len(Samples)-1):
        if invsig[i-1]*invsig[i+1]<0:
            zList.append(i)
        print(i)
    for z in zList:
        print('Zero = ', z)
        if abs(z-peakBin)<crossing:
            crossing=z
    return crossing#_left, cross_right

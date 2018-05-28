import numpy as np

#A = rdr.load_events('wave0.txt')
def inv(Samples):
    #B=bl.baseliner2(Samples)
    #Samples=Samples-B
    shift=10
    frac=0.60
    invLag=np.zeros(len(Samples))
    invLag[0:-shift]=-frac*Samples[shift:]
    invLead=np.zeros(len(Samples))
    invLead[shift:]=-frac*Samples[0:-shift]
    return Samples+invLag, Samples+invLead



def zerocrosser(inv1,inv2):
    cross_left = np.argmin(inv1)
    cross_right = cross_left+np.argmin(inv2[cross_left:-1])
    if cross_right<cross_left:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
    return cross_left, cross_right

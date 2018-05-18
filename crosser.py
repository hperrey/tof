import numpy as np

#A = rdr.load_events('wave0.txt')
def inv(Samples):
    #B=bl.baseliner2(Samples)
    #Samples=Samples-B
    shift=12
    frac=0.24
    invLag=np.zeros(len(Samples))
    invLag[0:-shift]=-frac*Samples[shift:]
    invLead=np.zeros(len(Samples))
    invLead[shift:]=-frac*Samples[0:-shift]
    return Samples+invLag, Samples+invLead

def zerocrosser(A1,A2):
    return np.argmin(A1), np.argmin(A2)


import matplotlib.pyplot as plt
import numpy as np
import Reader as rdr

#seems worse
def baseliner1(Samples):
    B=0
    H =np.histogram(Samples, max(Samples))
    for i in range(0,5):
        B+=i*H[0][i]
    B/=sum(H[0][0:5])
    return B

#seems better
def baseliner2(Samples):
    B=0
    for i in range(0,380):
        B+=Samples[i]
    B/=380
    return B

#A = rdr.load_events('wave0.txt')
#B1=baseliner1(A)
#print('B1 = ', B1)
#B2=baseliner2(A)
#print('B2 = ', B2)


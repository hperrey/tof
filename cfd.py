import matplotlib.pyplot as plt
import numpy as np
import Reader as rdr
import baseliner as bl

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



def zerocrosser(A1,A2):
    #flagLeft = 0
    #cl=0
    #flagRight=0
    #cr=0
    #rise = True
    #Left=0
    #Right=0
    # for i in range(3,len(A1)-3):
    #     if rise==True:
    #         if A1[i-1]*A1[i+1]<0 and A1[i-1]<0 and abs(A1[i+1])+abs(A1[i-1])>8:
    #             Left = i
    #             rise = False
    #         elif A1[i-2]*A1[i+2]<0 and A1[i-2]<0 and abs(A1[i+2])+abs(A1[i-2])>8:
    #             Left = i
    #             rise = False
    #         elif A1[i-3]*A1[i+3]<0 and A1[i-3]<0 and abs(A1[i+3])+abs(A1[i-3])>8:
    #             Left = i
    #             rise = False
        # if rise ==True:
        #     if A1[i-1]*A1[i+1]<0 and A1[i-1]<0 and A1[i+1]>0:
        #         if A1[i-2]<A1[i-1] and A1[i+2]>A1[i+1]:
        #             if A1[i-3]<A1[i-2] and A1[i+3]>A1[i+2]:
        #                   Left = i
        #                   rise = False
        #             else:
        #                 continue
        #         else:
        #             continue
        #     else:
        #         continue
        # elif rise == False:
        #     if A2[i-1]*A2[i+1]<0 and A2[i-1]>0 and A2[i+1]<0:
        #         if A2[i-2]>0 and A2[i+2]<0:
        #             if A2[i-3]>0 and A2[i+3]<0:
        #                 Right = i
        #                 break
        #         else:
        #             continue
        #     else:
        #         continue
    return np.argmin(A1), np.argmin(A2)

#A1,A2=inv(A.Samples[4344])
#Left,Right=zerocrosser(A1,A2)


#ALag, ALead=inv(A.Samples[9])
#plt.plot(ALag)
#plt.plot(ALead)
#plt.plot(A.Samples[9])
#plt.show()

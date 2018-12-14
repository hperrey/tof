import numpy as np
import pandas as pd
import tof
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import sys
sys.path.append('../analog_tof/')
import pyTagAnalysis as pta

def get_species(df, X=[0, 1000,1600, 4400], Y=[0, 0.4, 0.36, 0.34]):
    species=np.array([-1]*len(df), dtype=np.int8)
    #loop through pulses
    for n in range(0, len(df)):
        k = round(100*n/len(df))
        sys.stdout.write("\rGetting species %d%%" % k)
        sys.stdout.flush()
        #If we are to the left of the exclusion zone
        if df.qdc_det0[n] == 0:
            species[n] = -1
        elif df.qdc_det0[n]<X[1]:
            #inside exclusion box=>indistinguishable
            if (df.qdc_det0[n]-df.qdc_sg_det0[n])/N.qdc_det0[n] < Y[1]:
                species[n]=-1
                #above exclusion box=>neutron
            else:
                species[n]=1
        #If we are to the right of the exclusion zone
        #then loop through coordinates
        elif df.qdc_det0[n]>=X[1]:
            for i in range(1,len(X)):
                #find the interval the pulse belongs to
                if df.qdc_det0[n]<X[i]:
                    if X[i]>=X[1]:
                        #are we below(gamma) or above(neutron) of the discrimination line
                        if (df.qdc_det0[n]-df.qdc_sg_det0[n])/N.qdc_det0[n] < Y[i-1]+(df.qdc_det0[n]-X[i-1])*(Y[i]-Y[i-1])/(X[i]-X[i-1]):
                            species[n] = 0
                        else:
                            species[n] = 1
                        break
    df['species'] = species

def PSD(N, mode):
    if mode == "Analog":
        maxlg=4000
        N_dummy = N.query('0<qdc_det0<%d and 0.2<(qdc_det0-qdc_sg_det0)/qdc_det0<0.8'%maxlg)
        lg = N_dummy.qdc_det0
        ps = (N_dummy.qdc_det0-N_dummy.qdc_sg_det0)/N_dummy.qdc_det0
    elif mode == "Digital":
        maxlg=6000
        N_dummy = N.query('0<longgate<%d and 0<ps<0.4'%maxlg)
        lg = N_dummy.longgate
        ps = N_dummy.ps
    plt.hexbin(lg, ps, gridsize=50)
    plt.colorbar()
    plt.title('%s QDC v Pulseshape heatmap'%mode)
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/%sResults/psd_hexbin.png'%mode, format='png')
    plt.show()
    #2:KDE
    sns.kdeplot(lg, ps, n_levels=20, cmap="Greens", shade=True)
    plt.title('%s Kernel density estimation plot'%mode)
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/%sResults/psd_kde.png'%mode, format='png')
    plt.show()

    #3:Scatterplot
    if mode == "Analog":
        maxlg=4000
        #Rejected
        N_dummy = N.query('species==-1 and 0<qdc_det0<%d and 0.2<(qdc_det0-qdc_sg_det0)/qdc_det0<0.8'%maxlg)
        lg = N_dummy.qdc_det0
        ps = (N_dummy.qdc_det0-N_dummy.qdc_sg_det0)/N_dummy.qdc_det0
        plt.scatter(lg, ps, s=5, alpha=0.6, label = "Rejected")
        #Gammas
        N_dummy = N.query('species==0 and 0<qdc_det0<8000 and 0.2<(qdc_det0-qdc_sg_det0)/qdc_det0<0.8')
        lg = N_dummy.qdc_det0
        ps = (N_dummy.qdc_det0-N_dummy.qdc_sg_det0)/N_dummy.qdc_det0
        plt.scatter(lg, ps, s=5, alpha=0.6, label = "Gammas")
        #Neutrons
        N_dummy = N.query('species==1 and 0<qdc_det0<8000 and 0.2<(qdc_det0-qdc_sg_det0)/qdc_det0<0.8')
        lg = N_dummy.qdc_det0
        ps = (N_dummy.qdc_det0-N_dummy.qdc_sg_det0)/N_dummy.qdc_det0
        plt.scatter(lg, ps, s=5, alpha=0.6, label = "Neutrons")
    elif mode == "Digital":
        maxlg=6000
        #Rejected
        N_dummy = N.query('species==-1 and 0<longgate<%d and 0<ps<0.4'%maxlg)
        plt.scatter(N_dummy.longgate, N_dummy.ps, s=5, alpha=0.6, label = "Rejected")
        #Gammas
        N_dummy = N.query('species==0 and 0<longgate<%d and 0<ps<0.4'%maxlg)
        plt.scatter(N_dummy.longgate, N_dummy.ps, s=5, alpha=0.6, label = "Gammas")
        #Neutrons
        N_dummy = N.query('species==1 and 0<longgate<%d and 0<ps<0.4'%maxlg)
        plt.scatter(N_dummy.longgate, N_dummy.ps, s=5, alpha=0.6, label="Neutrons")
    plt.title('%s Pulse shape discrimination scatterplot'%mode)
    plt.legend()
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/%sResults/psd_scatter.png'%mode, format='png')
    plt.show()

def QDC(N, mode):
    if mode == "Analog":
        max_lg=15000
        #Sum
        N_dummy = N.query('0<qdc_det0<15000')
        lg = N_dummy.qdc_det0
        plt.hist(lg, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True)
    elif mode == "Digital":
        max_lg=15000
        plt.hist(N.longgate, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True)
    plt.title('%s QDC spectrum'%mode)
    plt.ylabel('counts')
    plt.xlabel('longgate')
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/%sResults/qdc_psd.png'%mode, format='png')
    plt.show()

def QDC_filtered(N, mode):
    if mode == "Analog":
        max_lg=15000
        #Sum
        N_dummy = N.query('0<qdc_det0<15000')
        lg = N_dummy.qdc_det0
        plt.hist(lg, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='Sum')
        #Rejected
        N_dummy = N.query('species==-1 and 0<qdc_det0<%s'%max_lg)
        lg = N_dummy.qdc_det0
        plt.hist(lg, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='Rejected')
        #Gammas
        N_dummy = N.query('species==0 and 0<qdc_det0<%s'%max_lg)
        lg = N_dummy.qdc_det0
        plt.hist(lg, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='Gammas')
        #Neutrons
        N_dummy = N.query('species==1 and 0<qdc_det0<%s'%max_lg)
        lg = N_dummy.qdc_det0
        plt.hist(lg, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='Neutrons')


    elif mode == "Digital":
        max_lg=8000
        plt.hist(N.longgate, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='sum')
        plt.hist(N.query('species==0').longgate, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='Gammas')
        plt.hist(N.query('species==1').longgate, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='Neutrons')
        plt.hist(N.query('species==-1').longgate, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='Rejected')
    plt.title('%s psd filtered qdc spectra'%mode)
    plt.ylabel('counts')
    plt.xlabel('longgate')
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/%sResults/qdc_filtered_psd.png'%mode, format='png')
    plt.legend()
    plt.show()

def ToF(N, mode):
    if mode == "Analog":
        left = 400
        right = 1100
        dt = N.tdc_det0_yap0
    elif mode == "Digital":
        left = 20
        right = 120
        dt=N.dt
    plt.hist(dt, bins=right-left, range=(left, right), histtype='step', linewidth=2)
    plt.title('%s Time of flight spectrum'%mode)
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/%sResults/tof_psd.png'%mode, format='png')
    plt.show()

def ToF_filtered(N, mode):
    if mode == "Analog":
        left = 400
        right = 1100
        #Sum
        dt = N.tdc_det0_yap0
        plt.hist(dt, range=(left, right), bins=(right-left), histtype='step', lw=2, label='Sum')
        #Gammas
        N_dummy = N.query('species==0')
        dt = N_dummy.tdc_det0_yap0
        plt.hist(dt, range=(left, right), bins=(right-left), histtype='step', lw=2, label='Gammas')
        #Neutrons
        N_dummy = N.query('species==1')
        dt = N_dummy.tdc_det0_yap0
        plt.hist(dt, range=(left, right), bins=(right-left), histtype='step', lw=2, label='Neutrons')
        #Rejected
        N_dummy = N.query('species==-1')
        dt = N_dummy.tdc_det0_yap0
        plt.hist(dt, range=(left, right), bins=(right-left), histtype='step', lw=2, label='Rejected')

    elif mode == "Digital":
        #ToF
        #Sum
        left = 20
        right = 120
        plt.hist(N.dt, bins=100, label='Sum', range=(left, right), histtype='step', linewidth=2)
        #Gammas
        plt.hist(N.query('species==0').dt, bins=100, label='Gammas', range=(left, right), histtype='step', linewidth=2)
        #Neutrons
        plt.hist(N.query('species==1').dt, bins=100, label='Neutrons', range=(left, right), histtype='step', linewidth=2)
        #Rejected
        plt.hist(N.query('species==-1').dt, bins=100, label='Rejected', range=(left, right), histtype='step', linewidth=2)
    plt.legend()
    plt.title('%s Time of flight spectrum'%mode)
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/%sResults/tof_filtered_psd.png'%mode, format='png')
    plt.show()

def ps_tof(N, mode):
    if mode == "Analog":
        N_dummy = N.query('%d<tdc_det0_yap0<%d and 0.2<(qdc_det0-qdc_sg_det0)/qdc_det0<0.7'%(500,800))
        dt = N_dummy.tdc_det0_yap0
        ps = (N_dummy.qdc_det0-N_dummy.qdc_sg_det0)/N_dummy.qdc_det0
        gs=150
    elif mode == "Digital":
        N_dummy = N.query('%d<dt<%d and 0<ps<0.4'%(20, 120))
        dt = N_dummy.dt
        ps=N_dummy.ps
        gs=50
    plt.hexbin(dt, ps, gridsize=gs, cmap="Reds")
    plt.colorbar()
    plt.title('%s Time of flight v pulseshape heatmap'%mode)
    plt.ylabel('PS')
    plt.xlabel('ToF')
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/%sResults/ps_tof.png'%mode, format='png')
    plt.show()




if __name__ == "__main__":

    while True:
        print("============================================================")
        print("||                     Greetings human                    ||")
        print("============================================================")
        print("Do you want to load analog or digital data (A/D)?")
        mode = input()
        if mode == "A" or mode == "a" or mode == "D" or mode == "d":
            break
        else:
            print("Invalid choice. Valid options are A and D")
    if mode == "A" or mode == "a":
        mode = "Analog"
        print("loading analog data")
        #N=pta.load_data('../analog_tof/Data1160_cooked.root')#.head(n=20000)
        N=pd.read_hdf('1160_species.h5')
        #print('Getting psd data')
        #get_species(N)

    if mode == "D" or mode == "d":
        mode = "Digital"
        print("loading and processing digitized data")
        N=pd.read_hdf('data/2018-12-13/N_cooked.h5')
        Y=pd.read_hdf('data/2018-12-13/Y_cooked.h5')
        #=====Get more info====#
        #tof.get_gates(N)
        #tof.get_species(N)
        tof.tof_spectrum(N, Y, tol_right = 2000)

    #==========generate plots============#
    def ask(q, fcn):
        print(q)
        answer = input()
        if answer == "Y" or answer == "y":
            print("use only a subset of dataset? (Y/N)")
            answer = input()
            if answer == "Y" or answer == "y":
                print("How many of the %d events do you wish to use?"%len(N))
                nrows = int(input())
            else:
                nrows = len(N)
            fcn(N.head(n=nrows), mode=mode)
    ask(q="Do you want to generate QDC-PS plots? (Y/N)", fcn=PSD)
    ask(q="Do you want to generate QDC spectrum? (Y/N)", fcn=QDC)
    ask(q="Do you want to generate PSD filtered QDC spectrum? (Y/N)", fcn=QDC_filtered)
    ask(q="Do you want to generate ToF spectrum? (Y/N)", fcn=ToF)
    ask(q="Do you want to generate PSD filtered ToF spectrum? (Y/N)", fcn=ToF_filtered)
    ask(q="Do you want to generate PS-ToF spectrum? (Y/N)", fcn=ps_tof)



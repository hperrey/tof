import pandas as pd
import tof
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)

def psd(N):
    #Make PSD plots
    N_dummy = N.query('0<longgate<8000 and 0<ps<0.4')
    #1:Hexbin
    hex = plt.hexbin(N_dummy.longgate, N_dummy.ps, gridsize=100)
    plt.show()
    #2:KDE
    sns.kdeplot(N_dummy.longgate, N_dummy.ps, n_levels=20, cmap="Greens", shade=True)
    plt.show()
    #3:Scatterplot
    N_dummy = N.query('species==-1 and 0<longgate<8000 and 0<ps<0.4')
    plt.scatter(N_dummy.longgate, N_dummy.ps, s=5, alpha=0.6, label = "Rejected")
    N_dummy = N.query('species==0 and 0<longgate<8000 and 0<ps<0.4')
    plt.scatter(N_dummy.longgate, N_dummy.ps, s=5, alpha=0.6, label = "Gammas")
    N_dummy = N.query('species==1 and 0<longgate<8000 and 0<ps<0.4')
    plt.scatter(N_dummy.longgate, N_dummy.ps, s=5, alpha=0.6, label="Neutrons")
    plt.legend()
    plt.show()

def QDC(N):
    #QDC
    max_lg=8000
    plt.hist(N.longgate, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='sum')
    plt.hist(N.query('species==0').longgate, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='Gammas')
    plt.hist(N.query('species==1').longgate, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='Neutrons')
    plt.hist(N.query('species==-1').longgate, range=(0, max_lg), bins=750, histtype='step', lw=2, log=True, label='Rejected')
    plt.title('psd filtered qdc spectra')
    plt.ylabel('counts')
    plt.xlabel('longgate')
    plt.legend()
    plt.show()

def ToF(N):
    #ToF
    #Sum
    print('sum!!!')
    left = 20
    right = 120
    tof.tof_spectrum(N, Y)
    plt.hist(N.dt, bins=100, label='Sum', range=(left, right), histtype='step', linewidth=2)
    #Gammas
    plt.hist(N.query('species==0').dt, bins=100, label='Gammas', range=(left, right), histtype='step', linewidth=2)
    #Neutrons
    plt.hist(N.query('species==1').dt, bins=100, label='Neutrons', range=(left, right), histtype='step', linewidth=2)
    #Rejected
    plt.hist(N.query('species==-1').dt, bins=100, label='Rejected', range=(left, right), histtype='step', linewidth=2)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #load and process data
    N=pd.read_hdf('data/2018-10-23/N2.h5').head(n=5000)
    Y=pd.read_hdf('data/2018-10-23/Y2.h5').head(n=5000)
    #Get Gates
    tof.get_gates(N)
    #Get species
    tof.get_species(N)

    #generate plots
    psd(N)
    QDC(N)
    ToF(N)

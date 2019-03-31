# coding: utf-8
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar

import seaborn as sns; sns.set(color_codes=True)
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import sys
sys.path.append('..')
sys.path.append('../../analog_tof')
import pyTagAnalysis as pta
from scipy.signal import convolve
import math

def get_fom(Emin=0, mode='digital'):
    if mode == 'digital':
        N=pd.read_parquet('../data/finalData/data1hour_clean.pq/', engine='pyarrow', columns=['qdc_lg', 'qdc_sg', 'amplitude','invalid', 'channel']).query('channel==0 and invalid==False and amplitude>40').reset_index().head(1000000)
    else:
        N=pta.load_data('../data/finalData/Data1793_cooked.root').head(1000000)
    if 'qdc_det0' in N:
        flg=0
        fsg=2.0#1.95#1.9
        N['qdc_lg'] = N.qdc_det0
        N['ps_new'] = ((flg*500+N.qdc_det0)-(fsg*60+N.qdc_sg_det0))/(flg*500+N.qdc_det0).astype(np.float64)
        Ecal = np.load('../data/finalData/Ecal_A.npy')
        N['E'] = Ecal[1] + Ecal[0]*N['qdc_lg']
        dummy=N.query('-1<ps_new<1 and %s<E<6 and 500<qdc_det0'%Emin)
        outpath ='/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/fom/FoM'

    else:
        fsg=4900#24500#25000
        flg=250
        #N['qdc_lg'] = N['qdc_lg_fine']
        #N['qdc_sg'] = N['qdc_sg_fine'] 
        N['ps_new'] = ((flg*500+N['qdc_lg'])-(fsg*60+N['qdc_sg']))/(flg*500+N['qdc_lg']).astype(np.float64)
        Ecal = np.load('../data/finalData/Ecal_D.npy')/1000
        N['E'] = Ecal[1] + Ecal[0]*N['qdc_lg']
        dummy=N.query('-1<ps_new<1 and %s<E<6'%Emin)
        outpath ='/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/fom/FoM'

    def gaus(x, a, x0, sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    k=70
    kernel = [0]*71
    a, x0, sigma = 1, 35, 3
    for i in range(0,70):
        kernel[i]=gaus(i+1, a, x0, sigma)
    kernel=np.array(kernel)
    kernel=kernel/sum(kernel)

    bins=1000
    H=np.histogram(dummy['ps_new'], bins=bins, range=(0,1))
    G=convolve(H[0], kernel, method='direct', mode='same')

    x=np.linspace((0)/bins,(bins-1)/bins,bins) 
    plt.figure(figsize=(6.2,4))
    #plot smoothed spectrum

    plt.plot(x, H[0], label='Pulse shape histogram')
    plt.plot(x, G,label='Smoothened pulse histogram')

    #peaks
    peaks = np.r_[True, G[1:] > G[:-1]] & np.r_[G[:-1] > G[1:], True]
    peaks[G < 0.1*max(G)]=False
    Plist=np.where(peaks)[0]

    valleys = np.r_[True, G[1:] < G[:-1]] & np.r_[G[:-1] < G[1:], True]
    valleys[0:Plist[0]]=False
    valleys[Plist[1]:-1]=False
    Vlist=np.where(valleys)[0]

    #plot extreme points
    print(Plist)
    print(Vlist)
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.axvline(Plist[0]/bins, ymin=0, ymax=G[Plist[0]]/ylim[1], lw=1.2, alpha=0.7, color='black')
    plt.axvline(Plist[1]/bins, ymin=0, ymax=G[Plist[1]]/ylim[1], lw=1.2, alpha=0.7, color='black')
    plt.axvline(Vlist[0]/bins, ymin=0, ymax=G[Vlist[0]]/ylim[1], label='extreme points', lw=1.2, alpha=0.7, color='black')


    #fit gaussian
    left, right, x0 =  0, Vlist[0], Plist[0]/1000
    x = H[1][left: right] -(H[1][1] - H[1][0])/2
    Gdummy = G[left:right]
    sigma = 0.2
    ymax=max(Gdummy)
    P1, C1 =  curve_fit(gaus, x, Gdummy, p0=[ymax, x0, sigma], bounds=( (ymax-1,x0-0.001, 0), (ymax+1, x0+0.001, 0.5) ))

    left, right, x0 = Vlist[0], 400, Plist[1]/1000
    x = (H[1][left: right] -(H[1][1] - H[1][0])/2)
    Gdummy = G[left:right]
    sigma = 0.5
    ymax=max(Gdummy)
    P2, C2 =  curve_fit(gaus, x, Gdummy, p0=[ymax, x0, sigma], bounds=( (ymax-1,x0-0.01, 0), (ymax+1, x0+0.01, 0.5) ))

    #fit_gaus(H=G, sigma=2)

    fwhm1 = 2*math.sqrt(2*math.log(2))*P1[2]
    fwhm2 = 2*math.sqrt(2*math.log(2))*P2[2]
    FoM= (P2[1]-P1[1])/(fwhm1+fwhm2)
    print('fom = ', FoM)
    x = np.linspace(0.0005,0.9995,1000)
    plt.plot(x, gaus(x, P1[0], P1[1], P1[2]), ms=1, label='Gaussian fit: FWHM = %s'%round(fwhm1, 2))
    plt.plot(x, gaus(x, P2[0], P2[1], P2[2]), ms=1, label='Gaussian fit: FWHM = %s'%round(fwhm2, 2))
    plt.xlim(0,0.7)
    plt.ylim(0,)
    plt.xlabel('Tail/total', fontsize=10)
    plt.ylabel('Counts', fontsize=10)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 10)
    plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

    #use the parameter to generate the psd spectrum
    plt.axes([.55, 0.65, .4, .3], facecolor='white')
    dummy=dummy.query('-0.1<ps_new<0.5')
    plt.hexbin((Ecal[1] + Ecal[0]*dummy.qdc_lg), dummy.ps_new, cmap='viridis', gridsize=(100,100), extent=[0,6,-0.1,0.5])
    plt.xlabel('MeV$_{ee}$', fontsize=10)
    plt.ylabel('Tail/total', fontsize=10)
    plt.yticks(rotation=90)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 10)
    plt.tight_layout()

    #plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/CompareResults/FOM_%s.pdf'%mode,format='pdf')
    #plt.show()
    plt.close()
    print('%s: '%mode, Vlist[0])
    return (FoM, Vlist[0])#, Plist[0], Plist[1], P1, P2

if __name__ == "__main__":
    #get_fom(Emin=0, mode='analog')
    #get_fom(Emin=0, mode='digital')

    Elist=np.linspace(0,3,31)
    FoM_digital = [0]*len(Elist)
    FoM_analog = [0]*len(Elist)
    Cut_digital = np.array([0]*len(Elist))
    Cut_analog = np.array([0]*len(Elist))
    i = 0
    for E in Elist:
        FoM_digital[i], Cut_digital[i] = get_fom(Emin=E, mode='digital')
        FoM_analog[i], Cut_analog[i] = get_fom(Emin=E, mode='analog')
        i += 1
    Cut_analog = Cut_analog/1000
    Cut_digital = Cut_digital/1000



    plt.figure(figsize=(6.2,3.1))
    plt.plot(Elist, FoM_digital, color='red', alpha=0.5, label='Figure of merit\nDigital setup')
    plt.scatter(Elist, FoM_digital, color='red', s=6)
    plt.plot(Elist, FoM_analog, color='blue', alpha=0.5, label='Figure of merit\nAnalog setup', linestyle='--')
    plt.scatter(Elist, FoM_analog, color='blue', s=6)
    plt.ylabel('Figure of merit', fontsize=10)
    plt.xlabel('Minimum energy deposition $MeV_{ee}$', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/CompareResults/PSD_comp.pdf',format='pdf')
    plt.show()

    plt.figure(figsize=(6.2,2.07))
    plt.plot(Elist, Cut_digital - Cut_digital[0], color='red', alpha=0.5, label='Digital setup')
    plt.scatter(Elist,  Cut_digital - Cut_digital[0], color='red', s=6)
    plt.plot(Elist,  Cut_analog - Cut_analog[0], color='blue', alpha=0.5, label='Analog setup', linestyle='--')
    plt.scatter(Elist,  Cut_analog - Cut_analog[0], color='blue', s=6)
    plt.ylabel('$\Delta Cut$', fontsize=10)
    plt.xlabel('Minimum energy deposition $MeV_{ee}$', fontsize=10)
    plt.ylim(-0.003, 0.003)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 10)
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/CompareResults/PSD_cut.pdf',format='pdf')
    plt.show()

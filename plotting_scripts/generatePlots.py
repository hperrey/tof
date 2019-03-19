# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns; sns.set(color_codes=True)
import sys
import dask.dataframe as dd
sys.path.append('../../analog_tof/')
sys.path.append('../tof')
import pyTagAnalysis as pta

#Load in dataframes
#Load the digitized
D = pd.read_parquet('../data/finalData/data1hour_clean.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'qdc_lg', 'qdc_sg', 'ps', 'tof', 'baseline_std']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<920')
D['qdc_lg'] = D['qdc_lg_fine']
D['qdc_sg'] = D['qdc_sg_fine']
D['ps'] = D['ps_fine']
fsg=3500
flg=230
D['ps'] = ((flg*500+D['qdc_lg_fine'])-(fsg*60+D['qdc_sg_fine']))/(flg*500+D['qdc_lg_fine']).astype(np.float64)
Dcal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/E_call_digi.npy')
Tshift_D = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_D.npy')
D['E'] = (D.qdc_lg*Dcal[0]+Dcal[1])/1000
D['tof'] = (D['tof'] - Tshift_D[1])/1000 + 3.3

#Load the analog
A = pta.load_data('../data/finalData/Data1793_cooked.root')
A['qdc_lg'] = A['qdc_det0']
A['qdc_sg'] = A['qdc_sg_det0']
flg=0
fsg=1.7
A['ps'] = ((flg*500+A.qdc_det0)-(fsg*60+A.qdc_sg_det0))/(flg*500+A.qdc_det0).astype(np.float64)
Acal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/E_call_analog.npy')
Tshift_A = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_A.npy')
Tcal = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/T_cal_analog.npy')
A['E'] = A.qdc_lg*Acal[0]+Acal[1]
A['tof'] = 1000 - A['tdc_det0_yap0']
A['tof'] = (A['tof'] - Tshift_A[1])*(-Tcal[0]) + 3.3

cmap='viridis'

def tof_hist(df, outpath, qdc_min, title, window=(0, 100000), fac=1, bins=150):
    dummy=df.query('%s<qdc_lg'%(qdc_min))
    plt.figure(figsize=(6,4))
    plt.hist(dummy.tof/fac, bins, range=window, histtype='step', lw=1.5)
    plt.xlabel('ToF(ns)', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title(title, fontsize=12)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
    plt.savefig(outpath+'tof.pdf', format='pdf')
    plt.show()

def tof_hist_filt(df, outpath, qdc_min, cut, title, window=(0, 100000), fac=1, bins=150):
    dummy=df.query('%s<qdc_lg'%(qdc_min))
    plt.figure(figsize=(6,4))
    plt.hist(dummy.tof/fac, bins, range=window, alpha=0.5, lw=1.5, label='Unfiltered')
    plt.hist(dummy.query('ps<%s'%cut).tof/fac, bins, range=window, histtype='step', lw=1.5, label='Gammas')
    plt.hist(dummy.query('ps>=%s'%cut).tof/fac, bins, range=window, histtype='step', lw=1.5, label='Neutrons')
    plt.xlabel('ToF(ns)', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title(title, fontsize=12)
    plt.legend()
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
    plt.savefig(outpath+'ToF_filt_CC.pdf', format='pdf')
    plt.show()

def psd(df, outpath, down, up, qdc_min, title):
    dummy=df.query('%s<ps<%s and E<6 and %s<qdc_lg'%(down, up, qdc_min))
    plt.figure(figsize=(6,4))
    plt.hexbin( dummy.E, dummy.ps, gridsize=(100, 100), cmap=cmap )
    plt.xlabel('Energy $MeV_{ee}$', fontsize=12)
    plt.ylabel('Tail/total', fontsize=12)
    plt.title(title, fontsize=12)
    ax = plt.gca()
    plt.colorbar()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
    plt.savefig(outpath+'psd.pdf', format='pdf')
    plt.show()

def tof_psd(df, outpath, fac, psdown, psup, todown, tofup, qdc_min, title):
    dummy=df.query('%s<ps<%s and %s<tof<%s and %s<qdc_lg'%(psdown, psup, todown, tofup, qdc_min))
    plt.figure(figsize=(6,4))
    plt.hexbin( dummy.tof/fac, dummy.ps, gridsize=(100, 100), cmap=cmap, norm=mc.LogNorm() )
    plt.xlabel('ToF(ns)', fontsize=12)
    plt.ylabel('Tail/total', fontsize=12)
    plt.title(title, fontsize=12)
    plt.colorbar()
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
    plt.savefig(outpath+'tof_psd.pdf', format='pdf')
    plt.show()

def qdc_hist(df, outpath, bins, window, title):
    plt.figure(figsize=(6,5))
    plt.hist(df.E, range=window, bins=500, log=True, histtype='step', lw=2)
    plt.xlabel('Energy ($MeV_{ee}$)\n ', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title(title, fontsize=12)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
    plt.savefig(outpath+'qdc.pdf', format='pdf')
    plt.show()

#tof_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', window=(0,150), fac=1, bins=150, qdc_min=0, title="Time of flight spectrum\nDigital setup")
#tof_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', window=(0,150), fac=1, bins=150, qdc_min=500, title="Time of flight spectrum\nAnalog setup")

# psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', down=0, up=0.4, qdc_min=0, title="Charge comparisson PSD spectrum\nDigital setup")
# psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', down=0.1, up=0.5, qdc_min=500, title="Charge comparisson PSD spectrum\nAnalog setup")

# tof_psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', fac=1, psdown=0, psup=1, todown=0, tofup=100, qdc_min=0, title="Tail/total and Time of flight\nDigital setup")
# tof_psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', fac=1, psdown=0, psup=1, todown=0, tofup=100, qdc_min=500, title="Tail/total and Time of flight\nAnalog setup")

tof_hist_filt(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', cut=0.17, window=(0,150), fac=1, bins=150, qdc_min=0, title="Filtered time of flight spectrum\nDigital setup")
tof_hist_filt(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', cut=0.27, window=(0,150), fac=1, bins=150, qdc_min=500, title="Filtered time of flight spectrum\nAnalog setup")


#qdc_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', bins=160, window=(0,16), title="Energy deposition spectrum\nAnalog setup")
#qdc_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', bins=80, window=(0,8), title="Energy deposition spectrum\nDigital setup")

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
D = pd.read_parquet('../data/finalData/data1hour.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'qdc_lg', 'qdc_sg', 'ps', 'tof']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<920')
D['qdc_lg'] = D['qdc_lg_fine']
D['qdc_sg'] = D['qdc_sg_fine']
D['ps'] = D['ps_fine']
Dcal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/E_call_digi.npy')
D['E'] = (D.qdc_lg*Dcal[0]+Dcal[1])/1000

#Load the analog
A = pta.load_data('../data/finalData/Data1793_cooked.root')
A['qdc_lg'] = A['qdc_det0']
A['qdc_sg'] = A['qdc_sg_det0']
A['ps'] = (A['qdc_det0']-A['qdc_sg_det0'])/A['qdc_det0']
Acal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/E_call_analog.npy')
A['E'] = A.qdc_lg*Acal[0]+Acal[1]
A['tof'] = A['tdc_det0_yap0']

def tof_hist(df, outpath, qdc_min, window=(0, 100000), fac=1, bins=150):
    dummy=df.query('%s<qdc_lg'%(qdc_min))
    plt.figure(figsize=(16,8))
    plt.hist(dummy.tof/fac, bins, range=window, histtype='step', lw=1.5)
    plt.xlabel('ToF(ns)', fontsize=16)
    plt.ylabel('Counts', fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 16)
    plt.savefig(outpath+'tof.pdf', format='pdf')
    plt.show()

def psd(df, outpath, down, up, qdc_min):
    dummy=df.query('%s<ps<%s and E<6 and %s<qdc_lg'%(down, up, qdc_min))
    plt.figure(figsize=(16,8))
    plt.hexbin( dummy.E, dummy.ps, gridsize=(100, 100), cmap='inferno' )
    plt.xlabel('Energy $MeV_{ee}$', fontsize=16)
    plt.ylabel('Tail/total', fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 16)
    plt.savefig(outpath+'psd.pdf', format='pdf')

def tof_psd(df, outpath, fac, psdown, psup, todown, tofup, qdc_min):
    dummy=df.query('%s<ps<%s and %s<tof<%s and %s<qdc_lg'%(psdown, psup, todown, tofup, qdc_min))
    plt.figure(figsize=(16,8))
    plt.hexbin( dummy.tof/fac, dummy.ps, gridsize=(100, 100), cmap='inferno', norm=mc.LogNorm() )
    plt.xlabel('ToF(ns)', fontsize=16)
    plt.ylabel('Tail/total', fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 16)
    plt.savefig(outpath+'tof_psd.pdf', format='pdf')

def qdc_hist(df, outpath, bins, window):
    plt.figure(figsize=(16,8))
    plt.hist(df.E, range=window, bins=500, log=True, histtype='step', lw=2)
    plt.xlabel('Energy ($MeV_{ee}$)', fontsize=16)
    plt.ylabel('Counts', fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 16)
    plt.savefig(outpath+'qdc.pdf', format='pdf')


#tof_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/',window=(0,150), fac=1000, bins=150, qdc_min=0)
tof_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', window=(350,650), fac=1, bins=300, qdc_min=500)

# psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', down=0.2, up=0.6, qdc_min=500)
# psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', down=0, up=0.4, qdc_min=0)

# tof_psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', fac=1, psdown=0, psup=1, todown=350, tofup=650, qdc_min=500)
# tof_psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', fac=1000, psdown=0, psup=1, todown=0, tofup=100000, qdc_min=0)

# qdc_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', bins=500, window=(0,16))
# qdc_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', bins=500, window=(0,8))

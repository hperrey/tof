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
D = pd.read_parquet('../data/finalData/data1hour_clean_dropped_samples_CNNpred.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'pred', 'qdc_lg', 'qdc_sg', 'ps', 'tof', 'baseline_std']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<610')
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

def tof_hist(df, outpath, tNmax, qdc_min, mode, window=(0, 100000), fac=1, bins=150):
    dummy=df.query('%s<qdc_lg'%(qdc_min))
    plt.figure(figsize=(6.2,3.1))
    plt.hist(dummy.tof/fac, bins, range=window, histtype='step', lw=1.5)
    plt.xlabel('ToF(ns)', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    #plt.title(title, fontsize=12)
    plt.ylim(0,1200)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
    textstr='%s'%mode
    plt.text(80, 280, textstr, fontsize=10, verticalalignment='top',bbox=dict(facecolor='none', edgecolor='blue', pad=0.5, boxstyle='square'))
    ax.annotate('Neutrons', xy=(35, 230), xytext=(20 ,600), fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('Gammas', xy=(5, 1000), xytext=(20,800), fontsize=10,
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    plt.axes([.52, .62, .45, .3], facecolor='white')
    dummy=df.query('0<tof<%s'%tNmax).reset_index()
    c=299792458
    m=939.565#*c
    x=1.055
    v=(x/(dummy.tof.astype(np.float64)*10**(-9)))
    vnat=v/c
    E=1/2*m*(vnat)**2
    plt.hist(E, range=(min(E),min(E)+6), bins=30, histtype='step', color='red', label='Neutron\nenergy', lw=1.5)
    plt.ylabel('Counts', fontsize=10)
    plt.xlabel('MeV', fontsize=10)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(outpath+'tof.pdf', format='pdf')
    plt.show()

def tof_hist_filt(df, outpath, qdc_min, cut, mode, CNN, window=(0, 100000), fac=1, bins=150):
    dummy=df.query('%s<qdc_lg'%(qdc_min))
    plt.figure(figsize=(6.2,3.1))
    plt.hist(dummy.tof/fac, bins, range=window, alpha=0.5, lw=1.5, label='Unfiltered')
    if CNN==True:
        plt.hist(dummy.query('pred<%s'%cut).tof/fac, bins, range=window, histtype='step', lw=1.5, label='Gammas')
        plt.hist(dummy.query('pred>=%s'%cut).tof/fac, bins, range=window, histtype='step', lw=1.5, label='Neutrons')
        outpath+='CNN'
    else:
        plt.hist(dummy.query('ps<%s'%cut).tof/fac, bins, range=window, histtype='step', lw=1.5, label='Gammas')
        plt.hist(dummy.query('ps>=%s'%cut).tof/fac, bins, range=window, histtype='step', lw=1.5, label='Neutrons')
    plt.xlabel('ToF(ns)', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    #plt.title(title, fontsize=12)
    plt.legend()
    plt.ylim(0,1200)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
    plt.tight_layout()
    textstr='%s'%mode
    plt.text(75, 400, textstr, fontsize=10, verticalalignment='top',bbox=dict(facecolor='none', edgecolor='blue', pad=0.5, boxstyle='square'))
    ax.annotate('Neutrons', xy=(45, 230), xytext=(55 ,600), fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('Gammas', xy=(5, 1000), xytext=(20,800), fontsize=10,
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    plt.savefig(outpath+'ToF_filt.pdf', format='pdf')
    plt.show()

def psd(df, outpath, CNN, down, cut, up, qdc_min, title, arrow1, arrow2, box):
    plt.figure(figsize=(6.2,3.1))
    if CNN==True:
        dummy=df.query('%s<pred<%s and E<6 and %s<qdc_lg'%(down, up, qdc_min))
        plt.hexbin( dummy.E, dummy.pred, gridsize=(100, 100), cmap=cmap )
        plt.ylabel('CNN prediction', fontsize=12)
        outpath+='CNN'
    else:
        dummy=df.query('%s<ps<%s and E<6 and %s<qdc_lg'%(down, up, qdc_min))
        plt.hexbin( dummy.E, dummy.ps, gridsize=(100, 100), cmap=cmap)
        plt.ylabel('Tail/total', fontsize=12)
    plt.xlabel('Energy $MeV_{ee}$', fontsize=12)
    #plt.title(title, fontsize=12)
    plt.axhline(y=cut, linestyle='--', color='white', lw=1)
    ax = plt.gca()
    textstr='%s'%title
    plt.text(box[0], box[1], textstr, fontsize=10, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    ax.annotate('2.23 $MeV$', xy=arrow1[0:2], xytext=arrow1[2:4], color='white', fontsize=10,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('4.44 $MeV$', xy=arrow2[0:2], xytext=arrow2[2:4], color='white', fontsize=10,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    plt.colorbar()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
    plt.tight_layout()
    plt.savefig(outpath+'psd.pdf', format='pdf')
    plt.show()

def tof_psd(df, outpath, fac, cut, psdown, psup, tofdown, tofup, qdc_min, title, txt_xy_neutron, txt_xy_gamma, arrow_xy_neutron, arrow_xy_gamma, CNN):
    plt.figure(figsize=(6.2,3.1))
    if CNN==True:
        dummy=df.query('%s<pred<%s and %s<tof<%s and %s<qdc_lg'%(psdown, psup, tofdown, tofup, qdc_min))
        plt.hexbin( dummy.tof/fac, dummy.pred, gridsize=(100, 100), cmap=cmap, norm=mc.LogNorm() )
        plt.ylabel('CNN prediction', fontsize=12)
        outpath+='CNN'
    else:
        dummy=df.query('%s<ps<%s and %s<tof<%s and %s<qdc_lg'%(psdown, psup, tofdown, tofup, qdc_min))
        plt.hexbin( dummy.tof/fac, dummy.ps, gridsize=(100, 100), cmap=cmap, norm=mc.LogNorm() )
        plt.ylabel('Tail/total', fontsize=12)
    plt.xlabel('ToF(ns)', fontsize=12)
    #plt.title(title, fontsize=12)
    plt.axhline(y=cut, linestyle='--', color='white', lw=1)
    plt.colorbar()
    ax = plt.gca()
    textstr='%s'%title
    plt.text(30, psdown+0.06, textstr, fontsize=10, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    ax.annotate('Neutrons', xy=arrow_xy_neutron, xytext=txt_xy_neutron, color='white', fontsize=10,
                arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('Gammas', xy=arrow_xy_gamma, xytext=txt_xy_gamma, color='white', fontsize=10,
                arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
    plt.tight_layout()
    plt.savefig(outpath+'tof_psd.pdf', format='pdf')
    plt.show()



def qdc_hist(df, outpath, bins, window, title):
    plt.figure(figsize=(6.2,3.1))
    plt.hist(df.E, range=window, bins=500, log=True, histtype='step', lw=2)
    plt.xlabel('Energy ($MeV_{ee}$)\n ', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title(title, fontsize=12)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
    plt.tight_layout()
    plt.savefig(outpath+'qdc.pdf', format='pdf')
    plt.show()

tof_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', window=(0,150), fac=1, bins=150, qdc_min=0, tNmax=65, mode="Time of flight spectrum\nDigital setup")
tof_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', window=(0,150), fac=1, bins=150, qdc_min=500, tNmax=50, mode="Time of flight spectrum\nAnalog setup")

# psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', CNN=False, cut=0.17, down=0, up=0.4, qdc_min=0, title="--- Discrimination cut: Digital setup", arrow1=[2, 0.08, 2.5, 0.01], arrow2=[4.2, 0.09, 4.7, 0.01], box=[2, 0.38])
# psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/',  CNN=True, cut=0.5, down=0, up=1, qdc_min=0, title="--- Discrimination cut: Digital setup", arrow1=[2, 0.25, 2.5, 0.4], arrow2=[4.2, 0.25, 4.7, 0.4], box=[2, 0.7])
# psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/',  CNN=False, cut=0.27, down=0.1, up=0.5, qdc_min=500, title="--- Discrimination cut: Analog setup", arrow1=[2, 0.2, 2.5, 0.11], arrow2=[4.2, 0.2, 4.7, 0.11], box=[2, 0.48])

# tof_psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', fac=1, psdown=-0.1, psup=0.5, tofdown=0, tofup=100, qdc_min=0, cut=0.17, title="--- Discrimination cut: Digital setup", txt_xy_gamma=[10, 0.4], txt_xy_neutron=[60, 0.4], arrow_xy_gamma=[4, 0.2], arrow_xy_neutron=[45, 0.3], CNN=False)
# tof_psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', fac=1, psdown=0, psup=1, tofdown=0, tofup=100, qdc_min=0, cut=0.5, title="--- Discrimination cut: Digital setup", txt_xy_gamma=[10, 0.4], txt_xy_neutron=[60, 0.7], arrow_xy_gamma=[4, 0.2], arrow_xy_neutron=[45, 0.85], CNN=True)
# tof_psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', fac=1, psdown=0, psup=1, tofdown=0, tofup=100, qdc_min=500, cut=0.27, title="--- Discrimination cut: Analog setup", txt_xy_gamma=[10, 0.48], txt_xy_neutron=[60, 0.48], arrow_xy_gamma=[4, 0.28], arrow_xy_neutron=[45, 0.38], CNN=False)

# tof_hist_filt(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', cut=0.17, window=(0,150), fac=1, bins=150, qdc_min=0, mode="Filtered time of flight spectrum\nCharge comparisson method\nDigital setup", CNN=False)
# tof_hist_filt(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', cut=0.5, window=(0,150), fac=1, bins=150, qdc_min=0, mode="Filtered time of flight spectrum\nCNN method\nDigital setup", CNN=True)
# tof_hist_filt(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', cut=0.27, window=(0,150), fac=1, bins=150, qdc_min=500, mode="Filtered time of flight spectrum\nCharge comparisson method\nAnalog setup", CNN=False)


# qdc_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', bins=160, window=(0,16), title="Energy deposition spectrum\nAnalog setup")
# qdc_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', bins=80, window=(0,8), title="Energy deposition spectrum\nDigital setup")

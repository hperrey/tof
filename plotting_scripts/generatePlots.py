# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.ticker as plticker
import seaborn as sns; sns.set(color_codes=True)
import sys
import dask.dataframe as dd
sys.path.append('../../analog_tof/')
sys.path.append('../tof')
import pyTagAnalysis as pta

#Load in dataframes
#Load the digitized
D = pd.read_parquet('../data/finalData/data1hour_CNN.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg', 'qdc_sg', 'ps', 'pred', 'tof', 'baseline_std']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<6100')
fsg=4900
flg=250
D['ps'] = ((flg*500+D['qdc_lg'])-(fsg*60+D['qdc_sg']))/(flg*500+D['qdc_lg']).astype(np.float64)
Dcal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Ecal_D.npy')
Tshift_D = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_D.npy')
D['E'] = (D.qdc_lg*Dcal[0]+Dcal[1])/1000
D['tof'] = (D['tof'] - Tshift_D[1])/1000 + 3.3

#Load the analog
A = pta.load_data('../data/finalData/Data1793_cooked.root')
A['qdc_lg'] = A['qdc_det0']
A['qdc_sg'] = A['qdc_sg_det0']
flg=0
fsg=2
A['ps'] = ((flg*500+A.qdc_det0)-(fsg*60+A.qdc_sg_det0))/(flg*500+A.qdc_det0).astype(np.float64)
Acal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Ecal_A.npy')
Tshift_A = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_A.npy')
Tcal = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/T_cal_analog.npy')
A['E'] = A.qdc_lg*Acal[0]+Acal[1]
A['tof'] = 1000 - A['tdc_det0_yap0']
A['tof'] = (A['tof'] - Tshift_A[1])*(-Tcal[0]) + 3.3

cmap='viridis'

def tof_hist(df, outpath, tNmax, qdc_min, mode, window, bins, fontsize, tnlow, tnhigh):
    dummy=df.query('%s<qdc_lg'%(qdc_min))
    plt.figure(figsize=(6.2,3.1))
    plt.hist(dummy.tof, bins, range=window, histtype='step', lw=1.5)
    plt.xlabel('ToF(ns)', fontsize= fontsize)
    plt.ylabel('Counts', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.ylim(0,1200)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    textstr='%s'%mode
    plt.text(80, 280, textstr, fontsize= fontsize, verticalalignment='top',bbox=dict(facecolor='none', edgecolor='blue', pad=0.5, boxstyle='square'))
    ax.annotate('Neutrons', xy=(35, 230), xytext=(20 ,600), fontsize= fontsize,
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('Gammas', xy=(5, 850), xytext=(20,700), fontsize= fontsize,
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    plt.axvline(x=tnlow, ymin=0, ymax=0.22, color='red', ls='--', lw=1)
    plt.axvline(x=tnhigh, ymin=0, ymax=0.22, color='red', ls='--', lw=1)
    plt.axes([.52, .62, .45, .3], facecolor='white')
    dummy=df.query('%s<tof<%s'%(tnlow, tnhigh)).reset_index()
    c=299792458
    m=939.565#*c
    x=1.055
    v=(x/(dummy.tof.astype(np.float64)*10**(-9)))
    vnat=v/c
    E=1/2*m*(vnat)**2
    vlow = (x/(tnhigh*10**(-9)))/c
    vhigh = (x/(tnlow*10**(-9)))/c
    Elow = 1/2*m*(vlow)**2
    Ehigh = 1/2*m*(vhigh)**2
    plt.hist(E, range=(Elow,Ehigh), bins=tnhigh-tnlow, histtype='step', color='red', label='Neutron\nenergy', lw=1.5)
    plt.ylabel('Counts', fontsize= fontsize)
    plt.xlabel('E(MeV)', fontsize= fontsize)
    plt.legend(fontsize= fontsize)
    loc = plticker.MultipleLocator(base=1.0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(loc)

    plt.tight_layout()
    plt.savefig(outpath+'tof.pdf', format='pdf')
    plt.show()

def tof_Edep_Eneutron(df, outpath, fontsize, title, tnlow, tnhigh):
    dummy=df.query('%s<tof<%s'%(tnlow, tnhigh)).reset_index()
    c=299792458
    m=939.565#*c
    x=1.055
    v=(x/(dummy.tof.astype(np.float64)*10**(-9)))
    vnat=v/c
    E=1/2*m*(vnat)**2
    vlow = (x/(tnhigh*10**(-9)))/c
    vhigh = (x/(tnlow*10**(-9)))/c
    Elow = 1/2*m*(vlow)**2
    Ehigh = 1/2*m*(vhigh)**2
    dummy['Eneutron'] = E

    plt.figure(figsize=(6.2,3.1))
    plt.hexbin(dummy.Eneutron, dummy.E, extent=(Elow, Ehigh, 0, 3.5), cmap='viridis', gridsize=(50))
    plt.xlabel('Neutron energy $MeV$', fontsize= fontsize)
    plt.ylabel('Deposited energy $MeV_{ee}$', fontsize= fontsize)
    ax = plt.gca()
    #textstr='%s'%title
    #plt.text(10, 5.5, textstr, fontsize= fontsize, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    plt.colorbar()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()
    plt.savefig(outpath+'tof_Edep_Eneutron.pdf', format='pdf')
    plt.show()

def tof_hist_filt(df, outpath, qdc_min, cut, mode, CNN, window, bins, fontsize):
    dummy=df.query('%s<qdc_lg'%(qdc_min))
    plt.figure(figsize=(6.2,2.8))
    plt.hist(dummy.tof, bins, range=window, alpha=0.5, lw=1.5, label='Unfiltered')
    if CNN==True:
        plt.hist(dummy.query('pred<%s'%cut).tof, bins, range=window, histtype='step', lw=1.5, label='Gammas')
        plt.hist(dummy.query('pred>=%s'%cut).tof, bins, range=window, histtype='step', lw=1.5, label='Neutrons')
        outpath+='CNN'
    else:
        plt.hist(dummy.query('ps<%s'%cut).tof, bins, range=window, histtype='step', lw=1.5, label='Gammas')
        plt.hist(dummy.query('ps>=%s'%cut).tof, bins, range=window, histtype='step', lw=1.5, label='Neutrons')
    plt.xlabel('ToF(ns)', fontsize= fontsize)
    plt.ylabel('Counts', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.legend()
    plt.ylim(0,1200)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()
    textstr='%s'%mode
    plt.text(75, 500, textstr, fontsize= fontsize, verticalalignment='top',bbox=dict(facecolor='none', edgecolor='blue', pad=0.5, boxstyle='square'))
    ax.annotate('Neutrons', xy=(45, 230), xytext=(55 ,600), fontsize= fontsize,
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('Gammas', xy=(5, 1000), xytext=(20,800), fontsize= fontsize,
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    plt.savefig(outpath+'ToF_filt.pdf', format='pdf')
    plt.show()

def psd(df, outpath, CNN, down, cut, up, qdc_min, title, arrow1, arrow2, box, fontsize):
    plt.figure(figsize=(6.2,3.1))
    if CNN==True:
        dummy=df.query('%s<pred<%s and E<6 and %s<qdc_lg'%(down, up, qdc_min))
        plt.hexbin( dummy.E, dummy.pred, gridsize=(100, 100), cmap=cmap, extent=(0,6, down, up))
        plt.ylabel('CNN prediction', fontsize= fontsize)
        outpath+='CNN'
    else:
        dummy=df.query('%s<ps<%s and E<6 and %s<qdc_lg'%(down, up, qdc_min))
        plt.hexbin( dummy.E, dummy.ps, gridsize=(100, 100), cmap=cmap, extent=(0,6, down, up))
        plt.ylabel('Tail/total', fontsize= fontsize)
    plt.xlabel('Energy $MeV_{ee}$', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.axhline(y=cut, linestyle='--', color='white', lw=1)
    ax = plt.gca()
    textstr='%s'%title
    plt.text(box[0], box[1], textstr, fontsize= fontsize, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    ax.annotate('2.23 $MeV$', xy=arrow1[0:2], xytext=arrow1[2:4], color='white', fontsize= fontsize,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('4.44 $MeV$', xy=arrow2[0:2], xytext=arrow2[2:4], color='white', fontsize= fontsize,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    plt.colorbar()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()
    plt.savefig(outpath+'psd.pdf', format='pdf')
    plt.show()

def tof_E(df, outpath, fontsize, title):
    plt.figure(figsize=(6.2,3.1))
    plt.hexbin(df.tof, df.E, extent=(0,100, 0, 6), norm=mc.LogNorm(), cmap='viridis')
    plt.xlabel('Time of flight(ns)', fontsize= fontsize)
    plt.ylabel('Energy $MeV_{ee}$', fontsize= fontsize)
    ax = plt.gca()
    #textstr='%s'%title
    #plt.text(10, 5.5, textstr, fontsize= fontsize, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    ax.annotate('Gammas', xy=(5,2), xytext=(10,3), color='white', fontsize= fontsize,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('Neutrons', xy=(45,2), xytext=(55,3), color='white', fontsize= fontsize,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    plt.colorbar()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()
    plt.savefig(outpath+'tof_E.pdf', format='pdf')
    plt.show()


def tof_psd(df, outpath, cut, psdown, psup, tofdown, tofup, qdc_min, title, txt_xy_neutron, txt_xy_gamma, arrow_xy_neutron, arrow_xy_gamma, CNN, fontsize):
    plt.figure(figsize=(6.2,3.1))
    if CNN==True:
        dummy=df.query('%s<pred<%s and %s<tof<%s and %s<qdc_lg'%(psdown, psup, tofdown, tofup, qdc_min))
        plt.hexbin( dummy.tof, dummy.pred, gridsize=(100, 100), cmap=cmap, norm=mc.LogNorm() )
        plt.ylabel('CNN prediction', fontsize= fontsize)
        outpath+='CNN'
    else:
        dummy=df.query('%s<ps<%s and %s<tof<%s and %s<qdc_lg'%(psdown, psup, tofdown, tofup, qdc_min))
        plt.hexbin( dummy.tof, dummy.ps, gridsize=(100, 100), cmap=cmap, norm=mc.LogNorm() )
        plt.ylabel('Tail/total', fontsize= fontsize)
    plt.xlabel('ToF(ns)', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.axhline(y=cut, linestyle='--', color='white', lw=1)
    plt.colorbar()
    ax = plt.gca()
    textstr='%s'%title
    plt.text(30, psdown+0.06, textstr, fontsize= fontsize, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    ax.annotate('Neutrons', xy=arrow_xy_neutron, xytext=txt_xy_neutron, color='white', fontsize= fontsize,
                arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('Gammas', xy=arrow_xy_gamma, xytext=txt_xy_gamma, color='white', fontsize= fontsize,
                arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()
    plt.savefig(outpath+'tof_psd.pdf', format='pdf')
    plt.show()



def qdc_hist(df, outpath, bins, window, title, fontsize):
    plt.figure(figsize=(6.2,3.1))
    plt.hist(df.E, range=window, bins=500, log=True, histtype='step', lw=2)
    plt.xlabel('Energy ($MeV_{ee}$)\n ', fontsize= fontsize)
    plt.ylabel('Counts', fontsize= fontsize)
    plt.title(title, fontsize= fontsize)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = fontsize)
    plt.tight_layout()
    plt.savefig(outpath+'qdc.pdf', format='pdf')
    plt.show()

fontsize = 10
#tof_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', window=(-20,130), bins=150, qdc_min=0, tNmax=75, fontsize=fontsize, mode="Time of flight spectrum\nDigital setup", tnlow=31, tnhigh=65)
#tof_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', window=(-20,130), bins=150, qdc_min=500, tNmax=50, fontsize=fontsize, mode="Time of flight spectrum\nAnalog setup", tnlow=28, tnhigh=50)

psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', CNN=False, cut=0.222, down=0, up=0.4, qdc_min=0, fontsize=fontsize, title="--- Discrimination cut: Digital setup", arrow1=[2, 0.08, 2.5, 0.01], arrow2=[4.2, 0.09, 4.7, 0.01], box=[2, 0.38])
#psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/',  CNN=True, cut=0.635, down=0, up=1, qdc_min=0, fontsize=fontsize, title="--- Discrimination cut: Digital setup", arrow1=[2, 0.25, 2.5, 0.4], arrow2=[4.2, 0.25, 4.7, 0.4], box=[2, 0.9])
#psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/',  CNN=False, cut=0.259, down=0.1, up=0.5, qdc_min=500, fontsize=fontsize, title="--- Discrimination cut: Analog setup", arrow1=[2, 0.2, 2.5, 0.11], arrow2=[4.2, 0.2, 4.7, 0.11], box=[2, 0.48])

tof_psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', psdown=-0.1, psup=0.5, tofdown=0, tofup=100, qdc_min=0, cut=0.222, fontsize=fontsize, title="--- Discrimination cut: Digital setup", txt_xy_gamma=[10, 0.3], txt_xy_neutron=[70, 0.3], arrow_xy_gamma=[5, 0.17], arrow_xy_neutron=[50, 0.2], CNN=False)
#tof_psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', psdown=0, psup=1, tofdown=0, tofup=100, qdc_min=0, cut=0.635, fontsize=fontsize, title="--- Discrimination cut: Digital setup", txt_xy_gamma=[10, 0.4], txt_xy_neutron=[60, 0.7], arrow_xy_gamma=[4, 0.2], arrow_xy_neutron=[45, 0.85], CNN=True)
#tof_psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', psdown=0, psup=1, tofdown=0, tofup=100, qdc_min=500, cut=0.259, fontsize=fontsize, title="--- Discrimination cut: Analog setup", txt_xy_gamma=[10, 0.48], txt_xy_neutron=[60, 0.48], arrow_xy_gamma=[4, 0.28], arrow_xy_neutron=[45, 0.38], CNN=False)

tof_hist_filt(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', cut=0.222, window=(-20,130), bins=150, qdc_min=0, fontsize=fontsize, mode="Filtered time of flight spectrum\nCharge comparisson method\nDigital setup", CNN=False)
#tof_hist_filt(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', cut=0.635, window=(-20,130), bins=150, qdc_min=0, fontsize=fontsize, mode="Filtered time of flight spectrum\nCNN method\nDigital setup", CNN=True)
#tof_hist_filt(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', cut=0.259, window=(-20,130), bins=150, qdc_min=500, fontsize=fontsize, mode="Filtered time of flight spectrum\nCharge comparisson method\nAnalog setup", CNN=False)

#tof_E(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', fontsize=12, title='Digital setup')
#tof_E(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', fontsize=12, title='Analog setup')

#tof_Edep_Eneutron(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', fontsize=12, title='Analog setup', tnlow=30, tnhigh=65)
#tof_Edep_Eneutron(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', fontsize=12, title='Analog setup', tnlow=25, tnhigh=50)

#qdc_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', bins=160, window=(0,16), fontsize=fontsize, title="Energy deposition spectrum\nAnalog setup")
#qdc_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', bins=80, window=(0,8), fontsize=fontsize, title="Energy deposition spectrum\nDigital setup")

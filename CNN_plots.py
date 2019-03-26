# coding: utf-8
import numpy as np
import os
#Pandas
import pandas as pd
#Dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
#Neural network stuff
import keras
import tof
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib.colors as mc
#D=pd.read_parquet('data/finalData/data1hour_clean_dropped_samples_CNNpred.pq/',engine='pyarrow')
D=pd.read_parquet('data/finalData/CNNtest.pq/',engine='pyarrow')

fsg=3500
flg=230
D['ps'] = ((flg*500+D['qdc_lg_fine'])-(fsg*60+D['qdc_sg_fine']))/(flg*500+D['qdc_lg_fine']).astype(np.float64)
Tshift_D = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_D.npy')
D['tof'] = (D['tof'] - Tshift_D[1])/1000 + 3.3


# #Time of Flight V prediction
# d=D.query('0 < tof < 100')
# plt.figure(figsize=(6,4))
# plt.hexbin(d.tof, d.pred, cmap='viridis')
# ax = plt.gca()
# ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
# plt.ylabel('CNN prediction', fontsize=12)
# plt.xlabel('Time of Flight(ns)', fontsize=12)
# plt.title('CNN prediction as a function of time of flight', fontsize=12)
# plt.colorbar()
# #plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/ToF_CNN_hex.pdf', format='pdf')
# plt.show()
# #Time of Flight
# d=D.query('0 < tof < 150')
# plt.figure(figsize=(6,4))
# plt.hist(d.tof, bins=100, range=(0,100), alpha=0.5)
# plt.hist(d.query('pred<0.5').tof, bins=150, range=(0,150), histtype='step', lw=2)
# plt.hist(d.query('pred>=0.5').tof, bins=150, range=(0,150), histtype='step', lw=2)
# ax = plt.gca()
# ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
# plt.ylabel('Counts', fontsize=12)
# plt.xlabel('Time of Flight(ns)', fontsize=12)
# plt.title('Time of Flight Spectrum', fontsize=12)
# #plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/ToF_filt_CNN.pdf', format='pdf')
# plt.show()

# #prediction
# plt.figure(figsize=(6,4))
# Ecal = np.load('data/finalData/E_call_digi.npy')/1000
# D['E'] = Ecal[1] + Ecal[0]*D['qdc_lg_fine']
# d=D.query('E<6')
# plt.hexbin(d.E, d.pred, cmap='viridis')
# ax = plt.gca()
# ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
# plt.ylabel('CNN Prediction', fontsize=12)
# plt.xlabel('Energy(MeV$_{ee}$)', fontsize=12)
# plt.title('CNN prediction as a function of deposited energy', fontsize=12)
# plt.colorbar()
# #plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/CNN_E.pdf', format='pdf')
# plt.show()


#PSD comp
dummy=D.query('-0.4<=ps<0.6')
H = sns.JointGrid(dummy.ps, dummy.pred)
H = H.plot_joint(plt.hexbin, cmap='viridis', gridsize=(100,100), norm=mc.LogNorm())
H.ax_joint.set_xlabel('Tail/total', fontsize=12)
H.ax_joint.set_ylabel('CNN prediction', fontsize=12)
_ = H.ax_marg_x.hist(dummy.ps, color="blue", alpha=.5, bins=np.arange(-0.4, 0.6, 0.01))
_ = H.ax_marg_y.hist(dummy.pred, color="blue", alpha=.5, orientation="horizontal", bins=np.arange(0, 1, 0.01))
plt.setp(H.ax_marg_x.get_yticklabels(), visible=True)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
plt.setp(H.ax_marg_y.get_xticklabels(), visible=True)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # shrink fig so cbar is visible
cbar_ax = H.fig.add_axes([0.92, 0.08, .02, 0.7])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
#plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/tailTotal_vs_cnnPred.pdf', format='pdf')
plt.show()

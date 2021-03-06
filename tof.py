import numpy as np
import os
#Pandas
import pandas as pd
#Dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
#Neural network stuff
import keras

def load_dataframe(filepath, in_memory=False, mode='standard', engine='pyarrow'):
    """This function lets you load in a parquet dataframe.
    \nParameters:
    \nfilepath: string, path to the parquet file you wish to read.
    \nin_memory: boolean, determines wether to load into memory with pandas or create a pointer to data on disk using dask
    \nmode: string, options: \'full\', \'standard\', \'reduced\':
    \'full\' loads all columns,
    \'standard\' leaves out only a few columns used for debugging and
    \'reduced\' leaves out debug columns as well as the memory heavy samples arrays.
    \nengine: string, name of the parquet library to use for the reading. Default is \'pyarrow\', alternative is \'fastparquet\'"""
    cols = ['window_width', 'channel', 'event_number', 'timestamp', 'fine_baseline_offset',
            'amplitude', 'peak_index''qdc_lg', 'qdc_sg', 'ps', 'cfd_trig_rise', 'tof', 'invalid', 'pred']

    #Select columns to load
    if mode == 'full':
        cols  = None
        print('reading in all columns')
    elif mode == 'standard':
        cols += ['samples']
        print('reading in standard columns')
    elif mode == 'reduced':
        print('reading in reduced dataframe, not containing waveforms')

    #Load data
    if in_memory:
        print('Reading into memory with pandas')
        df = pd.read_parquet(filepath, engine='pyarrow', columns=cols)
    else:
        print('Preparing pointer to data with dask.')
        df =dd.read_parquet(filepath, engine=engine, columns=cols)

    #Throw away invalid events unless mode 'full' was selected and drop the 'valid' column
    if mode != 'full':
        df = df[df['invalid']==False]
        df.drop('invalid', axis=1)

    return df



def cook_data(filepath, threshold, maxamp, Nchannel, Ychannel, outpath="",model_path="", CNN_window=300,  baseline_int_window=20, lg_baseline_offset=0, sg_baseline_offset=0, frac=0.3, lg=0, sg=0, cleanUp=False, blocksize=25*10**6, repatition_factor=16):
    """Uses dask to process the txt output of WaveDump on all available logical cores and return a simple dataframe in parquet format
    \nfilepath = Path to file. Use * as a wildcard to read multiple textfile: e.g. file*.txt, will read file1.txt, file2.txt, file3.txt, etc into the same dataframe.
    \nthreshold = Wavedump triggers on all channels when one channel triggers, so to throw away empty events we must reenforce the threshold.
    \nmaxamp = Max amplitude varies with the offset used in the wavedump config file. If a pulse reaches the maxAmp then we want to throw it away as it is likely to have some part cut off.
    \nNchannel, Ychannel = the channel numbers in which neutron and gamma detectors are placed
    \noutpath = The path where the resulting dataframe is stored.
    \nbaseline_integration_window =20 integer number of bins used in baseline determination.
    \nlg/sg_baseline_offset = the baseline offset we use when integrating pulses, in order to compensate for underflow, and in order to rotate or linearize psd spectrum
    \nfine_baseline_offset: The baseline is forced to be an integer. The non integer part is multiplied by 1000 and cast to an int for later use in pulse integration.
    \ncleanUp: Boolean, wether to write events that \'failed\' for various reason (cfd trig fail or wobbly baseline). These events will be a small fraction, provided a reasonable threshold was applied. By default this parameter is false since they can be filter out using query(\'invalid==False\'), and are useful for debugging and only take up a little space.
    \nfrac=0.3 = the fraction of peak amplitude used in the cfd algorithm.
    \nlg=200 = the width of the longgate integration window in nanoseconds,
    \nsg=22 = width of the shortgate integration window in nanoseconds,
    \nblocksize=25*10**6 = The amount of data in bytes that will be processed on each thread/logica core. Experiment to find a value that works for your machine specs. Likely it will be between 10 and 100 MB"""
    filesize = os.stat(filepath).st_size
    Nblocks = int(round(0.5+(filesize / blocksize / repatition_factor)))
    print('processing ', filesize, ' bytes. Will generate', Nblocks, ' blocks' )
    print('Generating lazy instructions.')

    #==================#
    # Read in the file #
    #==================#
    df = dd.read_csv(filepath, header=None, usecols=[0, 2, 3, 5, 7],
                     names=['window_width', 'channel', 'event_number', 'timestamp', 'samples'],
                     dtype={'window_width': np.int32,
                            'channel': np.int8,
                            'event_number': np.int64,
                            'timestamp': np.int64,
                            'samples': np.object},
                     blocksize=blocksize)

    #====================#
    # Format the samples #
    #====================#
    #first convert the string into an integer array. Then subtract the baseline.
    df['samples'] = df['samples'].str.split().apply(lambda x: np.array(x, dtype=np.int16), meta=df['samples'])
    df['samples'] = df['samples'].apply(lambda x: x - int(round(np.average(x[0:baseline_int_window]))), meta=df['samples'])
    #The baseline is forced to be an integer. The non integer part is multiplied by 1000 and cast to an int for later use in pulse integration.
    df['fine_baseline_offset'] = np.int16(0)
    df['fine_baseline_offset'] =  df.apply(lambda x: int(0.5 + 1000 * np.average( x.samples[0:baseline_int_window] ) ),
                                         meta=df['fine_baseline_offset'], axis=1)


    #====================================#
    # Get amplitude and location of peak #
    #====================================#
    df['amplitude'] = df['samples'].apply(lambda x: np.max(np.absolute(x)), meta=df['samples']).astype(np.int16)
    df['peak_index'] = df['samples'].apply(np.argmin, meta=df['samples']).astype(np.int16)

    #====================#
    # Pulse integrations #
    #====================#
    # offsetting each bin by a certain baseline offset is equivalent to adding the product
    # of the integration window and the baseline offset to the integration.
    df = pulse_integration(df, lg, sg)

    #=======================#
    # generate cfd triggers #
    #=======================#
    df['cfd_trig_rise'] = np.int32(0)
    df['cfd_trig_rise'] = df.apply(lambda x: cfd(x, frac=0.3), meta=df['cfd_trig_rise'], axis=1)

    #===================#
    # Handle bad events #
    #===================#
    #Throw away events whose amplitude is below the threshold
    #df = df[df['amplitude'] >= ch_thr_mask[df['channel']]]
    df = df[df['amplitude'] >= threshold]
    #And those whose amplitude is greater than the expected maximum amplitude (likely have their tops cut off)
    df['cutoff'] = False
    df['cutoff'] = df['cutoff'].where(df['amplitude']<maxamp, True)
    #df = df[maxamp > df['amplitude']]
    #and those whose baseline jitters too much.
    df['baseline_std'] = np.float64(0)
    df['baseline_std'] = df['samples'].apply(lambda x: np.std(x[0:baseline_int_window]), meta=df['baseline_std'])
    #df = df[df['baseline_std'] < 2]
    df['wobbly_baseline'] = False
    df['wobbly_baseline'] = df['wobbly_baseline'].where(df['baseline_std']<2, True)
    #and those where the cfd triggering failed.cfd_trig_rise = -1 implies error. This occurs if the first bin is
    #above the cfd trigger point. We also want to ensure that the cfd trigger happens after the baseline determination
    #and with enough bins following to allow proper lg integration.
    df['cfd_too_early'] = False
    df['cfd_too_early'] = df['cfd_too_early'].where(df['cfd_trig_rise']/1000>baseline_int_window, True)
    #df = df[baseline_int_window*1000 < df['cfd_trig_rise']] #ensure baseline int window
    df['cfd_too_late_lg'] = False
    df['cfd_too_late_lg'] = df['cfd_too_late_lg'].where(df['cfd_trig_rise']/1000<(df['window_width'] - lg), True)
    #df = df[df['cfd_trig_rise'] < 1000*(df['window_width']-lg)] # ensure lg integration window

    #===========================#
    #Time of Flight correlations#
    #===========================#
    shift = int( (Nchannel - Ychannel)/abs(Nchannel-Ychannel) )
    df = get_tof(df, Nchannel, Ychannel, shift)

    #=======================================#
    #Convolutional neural network prediction#
    #=======================================#
    df['cfd_too_late_CNN'] = False
    if (model_path):
        df = cnn_discrim(df, model_path, CNN_window)

    #General Goodness parameter
    df['invalid'] = df['cutoff'] | df['wobbly_baseline'] | df['cfd_too_early'] | df['cfd_too_late_lg'] | df['cfd_too_late_CNN']

    #Throw away or keep bad events? I recommend keeping bad events. They can be useful for debugging, and you can choose
    #not to load them later by choosing 'mode' in load_data_frame() function.
    if cleanUp == True:
        df = df[df['invalid'] == False]

    with ProgressBar():
        if (outpath):
            #repartition the dataframe into fewer (and larger) blocks
            df = df.repartition(npartitions=df.npartitions // repatition_factor)
            #save to disk
            print('Processing dataframe and saving to disk')
            df.to_parquet(outpath, engine='pyarrow', compression='snappy')
    return df

def cnn_discrim(df, model_path, CNN_window):
    #throw away events which don-t have enough bins after cfd trigger to perform cnn discrimination

    df['cfd_too_late_CNN'] = df['cfd_too_late_CNN'].where(df['cfd_trig_rise']/1000>(df['window_width'] - CNN_window), True)
    # ensure lg integration window
    #load the model
    model=keras.models.load_model('%s'%model_path)
    #Prepare the model to be run on the GPU (https://github.com/keras-team/keras/issues/6124)
    model._make_predict_function()
    pre_trig = 20
    df['pred'] = np.float32(-99)
    df['pred'] = df.apply(lambda x: model.predict(x.samples[int(0.5+x.cfd_trig_rise/1000)-pre_trig:int(0.5+x.cfd_trig_rise/1000)
                                                            +CNN_window-pre_trig].reshape(1,CNN_window,1).astype(np.float64))[0][0], axis=1, meta=df['pred'])
    return df


def get_tof(df, Nchannel, Ychannel, shift):
    """calculates time difference between events located shift rows apart.
    It then runs the check_tof_ch, which ensures that only events in the neutron detector,
    which are correlated to gamma detector event are accepted. All other events land in the quarantine bin -999999"""
    df['tof'] = np.int32(0)
    df['tof'] =  ( ( 8*1000*df['timestamp'] + df['cfd_trig_rise'] ) - ( 8*1000*df['timestamp'].shift(shift) + df['cfd_trig_rise'].shift(shift) ) ).fillna(-999999).astype(np.int32)
    #Only events in the neutron detector, which are matched to yap events are used. all other events are sent to a quarantine bin.
    df['tof_ch_shift'] = df['channel'].shift(shift).fillna(-1).astype(np.int8)
    df['tof'] = df.apply(lambda x: check_tof_ch(x, Nchannel, Ychannel), meta=df['tof'], axis=1)

    def check_tof_ch(row, Nchannel, Ychannel):
        if ( row['channel'] == Nchannel ):
            if ( row['tof_ch_shift'] == Ychannel ):
                return row['tof']
        return -999999

    return df

def pulse_integration(df, lg, sg, pre_trig=25, lg_baseline_offset=0, sg_baseline_offset=0):
    """lg and sg: gatelengths in nanoseconds
    \nlg_baseline_offset, sg_baseline_offset: offset values that can be used to linearize the pulseshape parameter
    \npretrig: number of nanoseconds before cfd trigger to commence pulse integration.
    \nAdditionally the fine baseline offset parameter will be used. See cook_data function docstring"""
    #Long gate
    df['qdc_lg'] = np.int32(0)
    df['qdc_lg_offset'] = lg*(df['fine_baseline_offset'] + lg_baseline_offset)
    df['qdc_lg'] = df.apply(lambda x: 1000*(-1)*(np.trapz(x['samples'][int(0.5+x['cfd_trig_rise']/1000)-pre_trig:int(0.5+x['cfd_trig_rise']/1000)+lg-pre_trig])),
                            meta=df['qdc_lg'], axis=1)
    df['qdc_lg'] = df['qdc_lg'] + df['qdc_lg_offset']
    df.drop('qdc_lg_offset', axis=1)
    #Short gate
    df['qdc_sg'] = np.int32(0)
    df['qdc_sg_offset'] = sg*(df['fine_baseline_offset'] + sg_baseline_offset)
    df['qdc_sg'] = df.apply(lambda x: 1000*(-1)*(np.trapz(x['samples'][int(0.5+x['cfd_trig_rise']/1000)-pre_trig:int(0.5+x['cfd_trig_rise']/1000)+sg-pre_trig])),
                            meta=df['qdc_sg'], axis=1)
    df['qdc_sg'] = df['qdc_sg'] + df['qdc_sg_offset']
    df.drop('qdc_sg_offset', axis=1)
    #Get the Tail/total ratio
    df['ps'] = (df['qdc_lg']-df['qdc_sg'])/df['qdc_lg']
    return df

def get_tof_array(df, Nchannel, Ychannel, i, tol):
    """generates an array of the times of flight.
    \ndf = the dataframe to get tof format
    \n Nchannel = The channel of the neutron detector
    \n Ychannel = The channel of the gamma detector
    \n i = an index used for generating the array. The index named index will be reset everytime a new partition is loaded so it can not be used. initially i must equal zero. function should be called like this i=0; Tarray, i = get_tof_array"""
    Ntdummy = 0
    Ytdummy = 0
    L = len(df)
    tofarray = np.array([-20000]*L, dtype=np.int32)
    for index, row in df.iterrows():
        if (row['channel'] == Ychannel):
            Ytdummy = row['timestamp']*1000*8 + row['cfd_trig_rise']
        if (row['channel'] == Nchannel):
            Nch_index = i
            Ntdummy = row['timestamp']*1000*8 + row['cfd_trig_rise']
        if (0 < Ntdummy-Ytdummy < tol ):
            tofarray[Nch_index] = Ntdummy-Ytdummy
        i += 1
    return tofarray, i


def cfd(row, frac):
    peak = row['samples'][row['peak_index']]
    samples=row['samples']
    rise_index = 0
    #find the cfd rise point
    for i in range( max(0, row['peak_index']-40), row['peak_index']):
        if samples[i] < frac * peak:
            rise_index = i
            break
        else:
            rise_index = 0
    #rise_index is the first bin after we crossed the threshold.
    #We want deltaSamples/deltaTime across riseindex and riseindex-1
    slope_rise = (samples[rise_index] - samples[rise_index-1])#divided by deltaT 1ns

    #slope < 0 is a sign of error. fx a pulse located in first few bins and already above threshold in bin 0.
    #it means that the first bin to rise above cfd threshold is equal to the previous bin, which means it is not the first.
    if slope_rise == 0:
        #print('\neventnumber=', row['event_number'],'\nslope == 0!!!!\nindex=', rise_index,'\n', samples[rise_index-5:rise_index+5])
        #print('Amplitude: ', row['amplitude'])
        tfine_rise = -1
    else:
        tfine_rise = 1000*(rise_index-1) + int(round(1000*(peak*frac-samples[rise_index-1])/slope_rise))
    return np.int32(tfine_rise)

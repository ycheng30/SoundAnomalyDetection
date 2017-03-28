
""" This script file includes all the necessary functions that are needed 
for audio signal processing and event detection"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
#from features import mfcc
#from features import logfbank
from numpy.lib import stride_tricks
import scipy.io.wavfile as wav


""" segment the origianl audio file into frames with certain window length """
# segmenting the origianl audio file 
# default window size = 400ms
# return a 2D array: each row = each segment (consist of time points)
def segment_signal(rawsignal, sample_rate, window=0.4):
    
    pts_segment = int(window*sample_rate) # time data point in the  window frame
    num_segment = int(np.floor(len(rawsignal)/pts_segment)) # number of segmented window frame from the original audio signal
    
    seg_signal = np.zeros(shape=(num_segment,pts_segment))
    
    row_index = 0
    for x in range(0, len(rawsignal), pts_segment ):
        if (len(rawsignal)-x) < pts_segment:
            break
        seg_signal[row_index,:] = rawsignal[x:x+pts_segment]
        row_index+=1
        
    return seg_signal
    

""" calculate the corresponding row (i.e. the sample point) of the new 2D array audio signal  """
def calc_2Dsignal_index(window, actual_time):
    # window in s
    # actual_time in min, absolute time stamp in the audio file, 31:20min
    actual_time = np.floor(actual_time) + (actual_time - np.floor(actual_time))*100/60
    return int(actual_time*60/window)


""" calculate the corresponding row/index (i.e. the sample point) of the new 2D array audio signal  """
""" and do the manual labeling """
def time_to_index(window, actual_time1, actual_time2, label_list, label):
    # window in s
    # actual_time in min, absolute time stamp in the audio file, 31:20min
    # label everything within that time frame as a specific lable/class n 
    # label_list is the label list
    # label is the actual class assignment (0-3)
    

    index1 = calc_2Dsignal_index(window, actual_time1)
    index2 = calc_2Dsignal_index(window, actual_time2)

    label_list[index1:index2+1] = label
    
    return label_list
    
    

    
""" sliding window operation over the 2D segmented signal, and get the window average and difference from the mean """  
""" operate on the time domain """
def sliding_window(signal, sample_rate, slide_window=3):
    # slide_window, size of the window (how many samples)
    # signal is 2D, already segmented into small frames
    # return the mean signal in the window, and the difference from the mean
    
    mean_signal = np.zeros([signal.shape[0]-2, signal.shape[1]])
    diff_signal = mean_signal
    
    for rowi in range(1, signal.shape[0]-1 ):
        mean_signal[rowi-1, :] = np.sum(signal[rowi-1:rowi+2,:],axis=0)/3 
        diff_signal[rowi-1, :] = np.abs(signal[rowi,:] - mean_signal[rowi-1,:]) # take the |difference|
        
    return mean_signal, diff_signal



""" sliding window operation over the 2D segmented signal, and get the window average and difference from the mean """  
""" operate on the frequency domain """
def sliding_window_spectra(spectra, sample_rate, slide_window=3):
    # slide_window, size of the window (how many datapoints)
    # spectrogram is 2D, based on time signal segmented into small frames
    # return the mean spectrogram in the window, and the difference from the mean
    # difference is in terms of absolute value (abs)
    
    mean_spectra = np.zeros([spectra.shape[0], spectra.shape[1], spectra.shape[-1]-2])
    diff_spectra = np.zeros([spectra.shape[0], spectra.shape[1], spectra.shape[-1]-2])
    
    for rowi in range(1, spectra.shape[-1]-1 ):
        mean_spectra[:,:, rowi-1] = np.sum(spectra[:,:,rowi-1:rowi+2],axis=2)/3 
        diff_spectra[:,:, rowi-1] = np.abs(spectra[:,:,rowi] - mean_spectra[:,:,rowi-1])
        
    return mean_spectra, diff_spectra




""" calculate the spectrogram on each data sample (row) and plot it, and save the plot if needed """
def plot_spectrogram_segmented(inputsignal, row_index, sample_rate, nperseg=96, mode='magnitude', plot_option=False): 

    # plotting the relative amplitude (decibel dB) 
    # input insignal is segmented 2D signal array
    # for label training data, label=0 is the background
    
    sig = inputsignal[row_index,:]
    f, t, Sxx = signal.spectrogram(sig, fs=sample_rate, nperseg = nperseg, mode = mode) # fs/sampling_rate: intrinsic signal property
    
    if plot_option:
        #plt.figure(figsize=(10,8))
        #normalize the dB
        #tt = [0:len(t)]*400/len(t)
        plt.pcolormesh(range(0, len(t)), f, 100*20.*np.log10(Sxx)/(20.*np.log10(max(Sxx.flatten()))), cmap='jet') # display scale 0-100 dB
        plt.colorbar()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time index')
        plt.xlim([0, len(t)])
        #plt.xlim([0, 400])
        plt.ylim([0, 23000])
        plt.clim(0,100)

    #return f, t, Sxx
    return Sxx
    


# plotting spectrogram
def plot_spectrogram(f, t, spectrogram, nperseg=96, mode='magnitude'): 

    # plotting the relative amplitude (decibel dB) 
    # input insignal is segmented 2D signal array
    # for label training data, label=0 is the background
        
    #normalize the dB
    plt.pcolormesh(range(0, len(t)), f, 100*20.*np.log10(spectrogram)/(20.*np.log10(max(spectrogram.flatten()))), cmap='jet') # display scale 0-100 dB    
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time index')
    plt.xlim([0, len(t)])
    plt.ylim([0, 23000])
    plt.clim(0,100)
    
    


""" calculate the spectrogram: """ 
def stft(sig, frameSize, overlapFac=0.5, window=np.hamming):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames) 


""" scale frequency axis logarithmically:   """
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs
    
    
""" plot spectrogram: """    
def plotstft(signal,row_index, samplerate, binsize=96, plotpath=None, colormap="jet"): # default binsize/frame size = 1024
    
    samples = signal[row_index,:]
    s = stft(samples, binsize)
        
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    
    #ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    ims = 100*20.*np.log10(abs(sshow))/(20.*np.log10(max(abs(sshow.flatten()))))
    
    timebins, freqbins = np.shape(ims)
    
    plt.figure(figsize=(10,8))

    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])
    plt.clim(0,100)


    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()
        
    plt.clf()
    
    return sshow, freqbins, timebins




""" draw a histogram of the spectrogram (2D) pixel/amplitude distribution  """
def hist_spectrogram(spectrogram, bins = 80, plot_option=False):
    
    if plot_option:
        hist_spec = plt.hist(20.*np.log10(np.abs((spectrogram.flatten()))), bins)
        plt.title("Spectrum Amplitude Distribution");
        plt.xlabel('Amplitude (dB)')
        plt.ylabel('Count')
        plt.xlim([0, 120]);
        plt.ylim([0, 650]);
        plt.show()
    
    # do not plot the histogram
    else:
        temp = spectrogram.flatten()
        temp = temp[temp != 0]    
        hist_spec = np.histogram(20.*np.log10(np.abs(temp)), bins)

    return hist_spec


""" draw a histogram of the filter bank coef (2D) pixel/amplitude distribution  """
def hist_filterbank(filterank, bins = 80, plot_option=False):
    
    if plot_option:
        hist_filter = plt.hist(np.abs((filterank.flatten())), bins)
        plt.title("Filter_bank Coefficient Distribution");
        #plt.xlabel('Amplitude (dB)')
        #plt.ylabel('Count')
        plt.xlim([0, 16]);
        plt.ylim([0, 120]);
        plt.show()
    
    # do not plot the histogram
    else:
        temp = filterank.flatten()
        temp = temp[temp != 0]    
        hist_filter = np.histogram(np.abs(temp), bins)
    
    return hist_filter



""" estimate the FWHM of spectrogram (2D) pixel/amplitude distribution from the histogram """
def FWHM_hist(hist_spec):
    # Y is the histogram
    # X is the bin
    X = hist_spec[-1] # amplitude, x axis (bin)
    Y = hist_spec[0] # count of each bin
    half_max = max(Y) / 2.
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    #find the left and right most indexes
    left_idx = np.where(d > 0)[0][0] 
    right_idx = np.where(d < 0)[0][-1]
    return X[right_idx] - X[left_idx] #return the difference (full width)


    
    
""" calculate the MFCC/Filter bank coefficient on each spectrogram (for each sample) and plot them """
'''def plot_mfcc(newsignal,row_index,sample_rate,plot_option=True,winlen=0.005,winstep=0.01,numcep=20,nfilt=24,nfft=512,lowfreq=0,highfreq=None,preemph=0.97, ceplifter=22,appendEnergy=True):
    
    sig = newsignal[row_index,:]
    
    mfcc_feat = mfcc(sig,sample_rate,winlen,winstep,numcep, nfilt,nfft,lowfreq,highfreq,preemph, ceplifter,appendEnergy)
    
    fbank_feat = logfbank(sig,sample_rate)
    
    if plot_option:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(np.abs(mfcc_feat)), origin="lower", aspect="auto", cmap='jet', interpolation="none")
        plt.colorbar()
        plt.clim(0,30)
        plt.xlabel("index")
        plt.ylabel("mfcc_coef")


        plt.subplot(1, 2, 2)
        plt.imshow(np.transpose(np.abs(fbank_feat)), origin="lower", aspect="auto", cmap='jet', interpolation="none")
        plt.colorbar()
        plt.clim(0,20)
        plt.xlabel("index")
        plt.ylabel('fbank_coef')

        plt.show()

    return mfcc_feat, fbank_feat
'''

""" prepare the dataframe for preditive modeling """
def prepare_dataframe(headers, inputdata, sample_option=True, samplefrac = 1/8):
# data is the raw input data (2D numpy array)
# In case of a highly imbalanced case, do the random subsampling on the class with the most instances
# ylabel_name is in str, the name of the labels/y value

    df = pd.DataFrame(inputdata, columns=headers)

    ylabel_name = headers[-1] # 

    df[ylabel_name] = df[ylabel_name].astype(int)

    if sample_option:
        sampled_df = df[df[ylabel_name]==0].sample(frac = samplefrac, replace=True)
        remains = df[df[ylabel_name]!=0]
        newdf = pd.concat([sampled_df, remains])
    else:
        newdf = df
        del df


    # add the time point/index into the dataframe
    newdf['time_order'] = newdf.index

    # need to sort the data point based on index, since it's window frame along time essentially.
    newdf.sort(['time_order'], inplace=True)

    newdf.drop('time_order', axis=1, inplace=True)


    return newdf               




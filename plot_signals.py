import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils.config import Config
from scipy import signal

raw_vs_preprocess = False
stft = False
find_min_max = True

Subject   = 2
Condition = 0
Trial     = 2
Channel   = 50
j_windows = 0
config    = Config()


if stft:
    ## loading 1D signal
    path = "./data/SF_Obs_MLdata.mat"
    f = h5py.File(path,'r')
    ref = f["data"][Condition][Subject]
    eeg = np.array(f[ref])
    eeg = eeg[Trial,j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,Channel]
    plt.figure(0)
    plt.plot(eeg, 'r.-')
    plt.xlabel("Index")
    plt.ylabel("EEG Signal")
    plt.savefig('./Figs/1D_signal_Sub_'+str(Subject)+'_Cond_'+str(Condition)+'_Trial_'+str(Trial)+'_Channel_'+str(Channel)+'.png')
    #plt.show()
    plt.close()

    f, t, Sxx = signal.stft(eeg, fs=config.fs, nperseg=config.nperseg, noverlap=config.noverlap)
    print(eeg.shape, f.shape, t.shape, Sxx.shape, config.nTimeBins)
    plt.figure(1)
    im = plt.pcolormesh(t, f[:config.freq_cut], np.abs(Sxx[:config.freq_cut,:]), shading='gouraud')
    plt.colorbar(im)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig('./Figs/2D_signal_Sub_'+str(Subject)+'_Cond_'+str(Condition)+'_Trial_'+str(Trial)+'_Channel_'+str(Channel)+'.png')
    plt.show()
    plt.close()

########### Pre-processed vs Raw data ###########

if raw_vs_preprocess:
    path = "./data/SF_Img_MLdata.mat"
    with h5py.File(path, 'r') as file:
        f = h5py.File(path,'r')
        ref = f["data"][Condition][Subject]
    preprocessed_data = np.array(f[ref])
    

    path = "./data/SF_Img_data_nopre.mat"
    with h5py.File(path, 'r') as file:
        f = h5py.File(path,'r')
        ref = f["data"][Condition][Subject]
    raw_data = np.array(f[ref])

    for channel in range(config.n_channels):
         pre_data_c = preprocessed_data[Trial,:,channel]
         raw_data_c = raw_data[Trial,:,channel]
     
         fig = plt.figure(1)
         axs = fig.subplots(2, 1)
         axs[0].plot(pre_data_c, 'b')
         axs[0].set_ylabel("Pre-Processed")
         axs[1].plot(raw_data_c, 'b')
         axs[1].set_ylabel("Raw")
         plt.savefig('./Figs/1d_raw_vs_preprocessed/'+str(Subject)+'_Cond_'+str(Condition)+'_Trial_'+str(Trial)+'_Channel_'+str(channel)+'.png')
         #plt.show()
         plt.close()

if find_min_max:
    global_min, global_max = 0, 0
    path = "./data/SF_Img_data_nopre.mat"
    with h5py.File(path, 'r') as file:
        f = h5py.File(path,'r')
        for condition in config.conditions:
            for subject in config.all_subjects:
                ref = f["data"][condition][subject]
                data = np.array(f[ref])
                if np.min(data) < global_min:
                    global_min = np.min(data)
                if np.max(data) > global_max:
                    global_max = np.max(data)

    print(global_min,global_max)
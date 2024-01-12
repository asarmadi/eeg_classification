import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils.config import Config
from utils.utils import scale, stockwell
from scipy import signal

raw_vs_preprocess = True
stft = False
find_min_max = False

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
         raw_data_c_s = scale(raw_data[Trial,:,:])
         raw_data_c_s = raw_data_c_s[:,channel]

         raw_data_c_stockwell   = stockwell(raw_data_c,config, True)
         raw_data_c_s_stockwell = stockwell(raw_data_c_s,config, True)
         pre_data_c_stockwell   = stockwell(pre_data_c,config, True)
         print(raw_data_c_s_stockwell.shape)

         extent = (0, config.n_timepoints, config.fmin, config.fmax)
     
         fig = plt.figure(1)
         axs = fig.subplots(3, 1)
         axs[0].plot(pre_data_c, 'b')
         axs[0].set_ylabel("Pre-Processed")
         axs[1].plot(raw_data_c, 'b')
         axs[1].set_ylabel("Raw")
         axs[2].plot(raw_data_c_s, 'b')
         axs[2].set_ylabel("Scaled Raw")
         fig.suptitle(f"Subject {Subject}, Trial {Trial}, Condition {Condition}, Channel {channel}")
         plt.savefig('./Figs/1d_raw_vs_preprocessed/'+str(Subject)+'_Cond_'+str(Condition)+'_Trial_'+str(Trial)+'_Channel_'+str(channel)+'.png')
         #plt.show()
         plt.close()

         fig = plt.figure(2)
         axs = fig.subplots(3, 1)
         axs[0].imshow(pre_data_c_stockwell, origin='lower', extent=extent, aspect="auto")
         axs[0].set_ylabel("Pre-Processed")
         axs[1].imshow(raw_data_c_stockwell, origin='lower', extent=extent, aspect="auto")
         axs[1].set_ylabel("Raw")
         axs[2].imshow(raw_data_c_s_stockwell, origin='lower', extent=extent, aspect="auto")
         axs[2].set_ylabel("Scaled Raw")
         fig.suptitle(f"Subject {Subject}, Trial {Trial}, Condition {Condition}, Channel {channel}")
         plt.savefig('./Figs/1d_raw_vs_preprocessed/stockwell'+str(Subject)+'_Cond_'+str(Condition)+'_Trial_'+str(Trial)+'_Channel_'+str(channel)+'.png')
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
import h5py
import matplotlib.pyplot as plt
import numpy as np
from config import *
from scipy import signal

Subject   = 2
Condition = 0
Trial     = 2
Channel   = 1
j_windows = 1
path = "./data/SF_Obs_MLdata.mat"
f = h5py.File(path,'r')
ref = f["data"][Condition][Subject]
eeg = np.array(f[ref])
eeg = eeg[Trial,j_windows*window_inc:j_windows*window_inc+window_len,Channel]

## loading 1D signal

plt.figure(0)
plt.plot(eeg, 'r.-')
plt.xlabel("Index")
plt.ylabel("EEG Signal")
plt.savefig('./Figs/1D_signal_Sub_'+str(Subject)+'_Cond_'+str(Condition)+'_Trial_'+str(Trial)+'_Channel_'+str(Channel)+'.png')
#plt.show()
plt.close()

f, t, Sxx = signal.stft(eeg, fs=2000, nperseg=1000, noverlap=950)
plt.figure(1)
plt.pcolormesh(t, f[:100], np.abs(Sxx[:100,:]), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig('./Figs/2D_signal_Sub_'+str(Subject)+'_Cond_'+str(Condition)+'_Trial_'+str(Trial)+'_Channel_'+str(Channel)+'.png')
plt.close()
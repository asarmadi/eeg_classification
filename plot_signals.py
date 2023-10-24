import h5py
import matplotlib.pyplot as plt
import numpy as np
from config import *
from scipy import signal

Subject   = 2
Condition = 0
Trial     = 2
Channel   = 2
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

f, t, Sxx = signal.stft(eeg, fs=1000, nperseg=500, noverlap=450, padded=False)
print(eeg.shape, f.shape, t.shape, Sxx.shape)
plt.figure(1)
plt.pcolormesh(t, f[:50], np.abs(Sxx[:50,:]), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig('./Figs/2D_signal_Sub_'+str(Subject)+'_Cond_'+str(Condition)+'_Trial_'+str(Trial)+'_Channel_'+str(Channel)+'.png')
#plt.show()
plt.close()

f_f, t_f, Sxx_f = signal.stft(eeg, fs=1000, nperseg=300, noverlap=250)
f_t, t_t, Sxx_t = signal.stft(eeg, fs=1000, nperseg=500, noverlap=450, padded=True)
print(eeg.shape, f.shape, t.shape, f_f[:20].shape, Sxx_t.shape)
fig = plt.figure(1)
axs = fig.subplots(2, 1)
axs[0].pcolormesh(t_t, f_t[:50], np.abs(Sxx_t[:50,:]), shading='gouraud')
axs[0].set_ylabel("Padded")
axs[1].pcolormesh(t_f, f_f[:20], np.abs(Sxx_f[:20,:]), shading='gouraud')
axs[1].set_ylabel("Not Padded")
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
plt.savefig('./Figs/2D_signal_Sub_'+str(Subject)+'_Cond_'+str(Condition)+'_Trial_'+str(Trial)+'_Channel_'+str(Channel)+'.png')
plt.show()
plt.close()
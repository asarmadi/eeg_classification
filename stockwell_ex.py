import h5py
import numpy as np
import matplotlib.pyplot as plt
from stockwell import st
from utils.config import Config
from scipy import signal

def Normalize(ave, std, x):
    return ((x-ave)/std)

def scale(X):
    min_des, max_des = -1, 1
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max_des - min_des) + min_des
    return X_scaled

Subject   = 2
Condition = 0
Trial     = 2
Channel   = 50
j_windows = 0
config    = Config()
path = "./data/SF_Obs_data_nopre.mat"
f = h5py.File(path,'r')
ref = f["data"][Condition][Subject]
eeg = np.array(f[ref])
eeg = eeg[Trial,j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,Channel]
mean, std  = np.mean(eeg, axis=0), np.std(eeg, axis=0)
eeg = Normalize(mean, std, eeg)
eeg = scale(eeg)


t = np.linspace(0, 1.7, len(eeg))

fmin = 0    # Hz
fmax = 100  # Hz
df = 1./(t[-1]-t[0])  # sampling step in frequency domain (Hz)
fmin_samples = int(fmin/df)
fmax_samples = int(fmax/df)
print(df, t[-1],t[0],fmin_samples,fmax_samples)
stock = st.st(eeg, fmin_samples, fmax_samples)
print(stock.shape)
extent = (t[0], t[-1], fmin, fmax)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, eeg)
ax[0].set(ylabel='amplitude')
ax[1].imshow(np.abs(stock), origin='lower', extent=extent)
ax[1].axis('tight')
ax[1].set(xlabel='time (s)', ylabel='frequency (Hz)')
plt.savefig("./Figs/stockwell.png")
plt.close()

f1, t1, Sxx = signal.stft(eeg, fs=config.fs, nperseg=config.nperseg, noverlap=config.noverlap)
fig, ax = plt.subplots(1, 1, sharex=True)
im = plt.pcolormesh(t1, f1[:config.freq_cut], np.abs(Sxx[:config.freq_cut,:]), shading='gouraud')
plt.colorbar(im)
plt.savefig("./Figs/stft.png")
plt.close()
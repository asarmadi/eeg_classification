import h5py
import numpy as np
import matplotlib.pyplot as plt
from stockwell import st
from utils.config import Config

Subject   = 2
Condition = 0
Trial     = 2
Channel   = 50
j_windows = 0
config    = Config()
path = "./data/SF_Obs_MLdata.mat"
f = h5py.File(path,'r')
ref = f["data"][Condition][Subject]
eeg = np.array(f[ref])
eeg = eeg[Trial,j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,Channel]


t = np.linspace(0, 10, len(eeg))

fmin = 0  # Hz
fmax = 20  # Hz
df = 1./(t[-1]-t[0])  # sampling step in frequency domain (Hz)
fmin_samples = int(fmin/df)
fmax_samples = int(fmax/df)
stock = st.st(eeg, fmin_samples, fmax_samples)
print(stock.shape, np.abs(stock).shape)
extent = (t[0], t[-1], fmin, fmax)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, eeg)
ax[0].set(ylabel='amplitude')
ax[1].imshow(np.abs(stock), origin='lower', extent=extent)
ax[1].axis('tight')
ax[1].set(xlabel='time (s)', ylabel='frequency (Hz)')
plt.show()
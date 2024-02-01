import pywt
import h5py
import numpy as np
from utils.config import Config
import matplotlib.pyplot as plt

Subject   = 2
Condition = 0
Trial     = 2
Channel   = 50
j_windows = 0
level = 6
path = "./data/SF_Img_MLdata.mat"
config    = Config()
f = h5py.File(path,'r')
ref = f["data"][Condition][Subject]
eeg = np.array(f[ref])
eeg = eeg[Trial,j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,Channel]
# WPD tree
wptree = pywt.WaveletPacket(data=eeg, wavelet='db5', mode='symmetric', maxlevel=level)
levels = wptree.get_level(level, order = "freq")            

#Feature extraction for each node
features = []        
for node in levels:
    data_wp = node.data
    # Features group
    features.append(data_wp)

features = np.array(features)    


wavelet = 'db2'
level = 5
order = "freq"  # other option is "normal"
interpolation = 'nearest'
cmap = plt.cm.cool

# Construct wavelet packet
wp = pywt.WaveletPacket(eeg, wavelet, 'symmetric', maxlevel=level)
nodes = wp.get_level(level, order=order)
labels = [n.path for n in nodes]
values = np.array([n.data for n in nodes], 'd')
values = abs(values)

print(values.shape)

# Show signal and wavelet packet coefficients
fig = plt.figure()
fig.subplots_adjust(hspace=0.2, bottom=.03, left=.07, right=.97, top=.92)
ax = fig.add_subplot(2, 1, 1)
ax.set_title("signal")
ax.plot(eeg, 'b')

ax = fig.add_subplot(2, 1, 2)
ax.set_title("Wavelet packet coefficients at level %d" % level)
ax.imshow(values[:10], interpolation=interpolation, cmap=cmap, aspect="auto",
          origin="lower", extent=[0, 1, 0, len(values[:10])])
#ax.set_yticks(np.arange(0.5, len(labels) + 0.5), labels)

plt.show()
exit(0)
# Show spectrogram and wavelet packet coefficients
fig2 = plt.figure()
ax2 = fig2.add_subplot(211)
ax2.specgram(eeg, NFFT=64, noverlap=32, Fs=2, cmap=cmap,
             interpolation='bilinear')
ax2.set_title("Spectrogram of signal")
ax3 = fig2.add_subplot(212)
ax3.imshow(values, origin='upper', extent=[-1, 1, -1, 1],
           interpolation='nearest')
ax3.set_title("Wavelet packet coefficients")


plt.show()



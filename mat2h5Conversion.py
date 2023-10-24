import h5py
from config import *
from scipy import signal

def spectrogram_per_channel(g):
    channels = []
    for j in range(n_channels):
        f, t, Sxx = signal.stft(g[:, j], fs=1000, nperseg=300, noverlap=250)
        channels.append(np.abs(Sxx[:20,:]))
    return np.array(channels)

def Normalize(ave, std, x):
    return ((x-ave)/std)

n_samples_train = len(train_subjects)*n_windows*n_trial*2
print(n_samples_train, len(train_subjects), n_windows, n_trial)
n_samples_valid = len(valid_subjects)*n_windows*n_trial*2
n_samples_test  = len(test_subjects) *n_windows*n_trial*2

train_shape = (n_samples_train, n_channels, 20, 21)
valid_shape = (n_samples_valid, n_channels, 20, 21)
test_shape  = (n_samples_test,  n_channels, 20, 21)

f_train = h5py.File(file_path+'train_2d.h5', "w")
f_train.create_dataset("data", train_shape)
f_train.create_dataset("label", (n_samples_train,))
f_train.create_dataset("subject", (n_samples_train,))
f_train.create_dataset("trial", (n_samples_train,))

f_test = h5py.File(file_path+'test_2d.h5', "w")
f_test.create_dataset("data", test_shape)
f_test.create_dataset("label", (n_samples_test,))
f_test.create_dataset("subject", (n_samples_test,))
f_test.create_dataset("trial", (n_samples_test,))


f_valid = h5py.File(file_path+'valid_2d.h5', "w")
f_valid.create_dataset("data", valid_shape)
f_valid.create_dataset("label", (n_samples_valid,))
f_valid.create_dataset("subject", (n_samples_valid,))
f_valid.create_dataset("trial", (n_samples_valid,))

u = 0
s = 0    # sth sample in test set
v = 0    # vth sample in validation set

path = file_path + "SF_Obs_MLdata.mat"

f = h5py.File(path,'r')

train_eeg = []
for subject in train_subjects:
    for condition in range(2): # 0, 1 correpond to Flex First, and Extend First
        print(f'subject#: {subject}, condition: {condition}')
        ref = f["data"][condition][subject]
        eeg = np.array(f[ref])
        train_eeg.append(eeg)
#
train_eeg = np.array(train_eeg)
train_eeg = np.mean(train_eeg, axis=0)
mean, std = np.mean(train_eeg, axis=0), np.std(train_eeg, axis=0)
print(mean.shape)

for subject in train_subjects:
    for condition in range(2): # 0, 1 correpond to Flex First, and Extend First
        print(f'Train subject#: {subject}, condition: {condition}')
        ref = f["data"][condition][subject]
        eeg = np.array(f[ref])
        eeg = Normalize(mean, std, eeg)
        for i_trial in range(n_trial):
            for j_windows in range(n_windows):
                Sxx = spectrogram_per_channel(eeg[i_trial,j_windows*window_inc:j_windows*window_inc+window_len,:])
                f_train["data"][u,...] = Sxx
                f_train["subject"][u]  = subject
                f_train["label"][u]    = condition
                f_train["trial"][u]    = i_trial
                u += 1

for subject in test_subjects:
    for condition in range(2): # 0, 1 correpond to Flex First, and Extend First
        print(f'Test subject#: {subject}, condition: {condition}')
        ref = f["data"][condition][subject]
        eeg = np.array(f[ref])
        eeg = Normalize(mean, std, eeg)
        for i_trial in range(n_trial):
            for j_windows in range(n_windows):
                Sxx = spectrogram_per_channel(eeg[i_trial,j_windows*window_inc:j_windows*window_inc+window_len,:])
                f_test["data"][s,...] = Sxx
                f_test["subject"][s]  = subject
                f_test["label"][s]    = condition
                f_test["trial"][s]    = i_trial
                s += 1

for subject in valid_subjects:
    for condition in range(2): # 0, 1 correpond to Flex First, and Extend First
        print(f'Valid subject#: {subject}, condition: {condition}')
        ref = f["data"][condition][subject]
        eeg = np.array(f[ref])
        eeg = Normalize(mean, std, eeg)
        for i_trial in range(n_trial):
            for j_windows in range(n_windows):
                Sxx = spectrogram_per_channel(eeg[i_trial,j_windows*window_inc:j_windows*window_inc+window_len,:])
                f_valid["data"][v,...] = Sxx
                f_valid["subject"][v]  = subject
                f_valid["label"][v]    = condition
                f_valid["trial"][v]    = i_trial
                v += 1

f_train.close()
f_test.close()
f_valid.close()


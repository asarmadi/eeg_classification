import numpy as np

from braindecode.datasets import BCICompetitionIVDataset4

subject_id = 1
dataset = BCICompetitionIVDataset4(subject_ids=[subject_id])

from braindecode.preprocessing import (Preprocessor,
                                       exponential_moving_standardize,
                                       preprocess)

low_cut_hz = 1.  # low cut frequency for filtering
high_cut_hz = 200.  # high cut frequency for filtering, for ECoG higher than for EEG
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocess(dataset, [Preprocessor('crop', tmin=0, tmax=30)])

preprocessors = [
    Preprocessor('pick_types', ecog=True, misc=True),
    Preprocessor(lambda x: x / 1e6, picks='ecog'),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size, picks='ecog')
]

# Transform the data
preprocess(dataset, preprocessors)

# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Extract target sampling frequency
target_sfreq = dataset.datasets[0].raw.info['temp']['target_sfreq']

from braindecode.preprocessing import create_windows_from_target_channels

windows_dataset = create_windows_from_target_channels(
    dataset,
    window_size_samples=1000,
    preload=False,
    last_target_only=True
)

windows_dataset.target_transform = lambda x: x[0: 1]

subsets = windows_dataset.split('session')
train_set = subsets['train']
test_set = subsets['test']

import torch
from sklearn.model_selection import train_test_split

idx_train, idx_valid = train_test_split(np.arange(len(train_set)),
                                        random_state=100,
                                        test_size=0.2,
                                        shuffle=False)

valid_set = torch.utils.data.Subset(train_set, idx_valid)
train_set = torch.utils.data.Subset(train_set, idx_train)

from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
# Set random seed to be able to roughly reproduce results
# Note that with cudnn benchmark set to True, GPU indeterminism
# may still make results substantially different between runs.
# To obtain more consistent results at the cost of increased computation time,
# you can set `cudnn_benchmark=False` in `set_random_seeds`
# or remove `torch.backends.cudnn.benchmark = True`
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

#trainloader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True,  num_workers=4, pin_memory=False)
#for batch_idx, (inputs, targets, subjects) in enumerate(trainloader):
#    print(inputs.shape,targets.shape)
#    input('enter')
#exit(0)
n_out_chans = train_set[0][1].shape[0]
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = 1000  # 1 second long windows

model = ShallowFBCSPNet(
    n_chans,
    n_out_chans,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

# Send model to GPU
if cuda:
    model.cuda()

from skorch.callbacks import EpochScoring, LRScheduler
from skorch.helper import predefined_split
from mne import set_log_level

from braindecode import EEGRegressor

# These values we found good for shallow network for EEG MI decoding:
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 2


# Function to compute Pearson correlation coefficient
def pearson_r_score(net, dataset, y):
    preds = net.predict(dataset)
    corr_coeffs = []
    for i in range(y.shape[1]):
        corr_coeffs.append(np.corrcoef(y[:, i], preds[:, i])[0, 1])
    return np.mean(corr_coeffs)


regressor = EEGRegressor(
    model,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation,
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        'r2',
        ('valid_pearson_r', EpochScoring(pearson_r_score, lower_is_better=False, on_train=False,
                                         name='valid_pearson_r')),
        ('train_pearson_r', EpochScoring(pearson_r_score, lower_is_better=False, on_train=True,
                                         name='train_pearson_r')),
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
set_log_level(verbose='WARNING')

regressor.fit(train_set, y=None, epochs=n_epochs)

preds_test = regressor.predict(test_set)
y_test = np.stack([data[1] for data in test_set])
preds_train = regressor.predict(train_set)
y_train = np.stack([data[1] for data in train_set])
preds_valid = regressor.predict(valid_set)
y_valid = np.stack([data[1] for data in valid_set])

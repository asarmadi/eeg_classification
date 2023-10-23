import numpy as np

n_subjects   = 20
n_conditions = 4   # FF, EF, FT, ET
n_channels   = 56
n_timepoints = 2000
n_trial      = 36  # For Imaginary 

file_path    = './data/'  # Path to save the h5 files

window_len  = 1000
window_inc   = 10

train_trials = np.arange(0,30)
valid_trials = [30, 33, 35]
test_trials  = [31, 32, 34]

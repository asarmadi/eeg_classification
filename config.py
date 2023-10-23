import numpy as np

n_subjects   = 20
n_conditions = 4   # FF, EF, FT, ET
n_channels   = 56
n_timepoints = 2000
n_trial      = 36  # For Imaginary 

file_path    = './data/'  # Path to save the h5 files

window_len  = 1000
window_inc  = 10
n_windows   = ((n_timepoints-window_len)//window_inc+1)

# Bad Subjects: [0,5,10,11,18]
#train_subjects = [1,2,3,4,6,8,9,12,13,14,15,16,17,19]
train_subjects  = [1,2,3,4,6]
valid_subjects  = [7]
test_subjects   = [9]

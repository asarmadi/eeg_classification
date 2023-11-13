import numpy as np

class Config:
      def __init__(self, model=''):
          self.model_type   = model
          self.n_subjects   = 20
          self.n_conditions = 4   # FF, EF, FT, ET
          self.n_channels   = 56
          self.n_timepoints = 2000
          self.n_trial      = 36  # For Imaginary it should be 22
          self.file_path    = './data/'  # Path to save the h5 files
          self.n_classes    = 1
          self.preprocess_normalize = False

          # Each segement
          self.window_len  = 1800
          self.window_inc  = 10
          self.n_windows   = ((self.n_timepoints-self.window_len)// self.window_inc+1)

          # STFT Hyper-parameters
          self.fs          = 1000
          self.nperseg     = 1000
          self.noverlap    = 990
          self.freq_cut    = 101
          self.nTimeBins   = 2*(self.window_len - self.nperseg) // (self.nperseg - self.noverlap) + 1
    
          # Bad Subjects: [0,5,10,11,18]
          self.train_subjects  = []
          self.all_subjects    = np.array([1,2,3,4,6,7,8,9,12,13,14,15,16,17,19])
          self.valid_subjects  = []
          self.test_subjects   = []
          self.conditions      = [0,1]

          if model == 'capsnet':
            # CNN (cnn)
            self.cnn_in_channels  = self.n_channels
            self.cnn_out_channels = 256
            self.cnn_kernel_size  = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels  = self.cnn_out_channels
            self.pc_out_channels = 32
            self.pc_kernel_size  = 9
            self.pc_num_routes   = 32 * 5 * 5

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = self.pc_num_routes
            self.dc_in_channels = 1
            self.dc_out_channels = 128

            # Decoder
            self.input_width  = self.nTimeBins
            self.input_height = self.freq_cut

          if 'lstm' in model:
            self.hidden_dim = 256
            self.layer_dim  = 2

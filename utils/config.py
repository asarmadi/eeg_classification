import numpy as np

class Config:
      def __init__(self, model=''):
          self.model_type   = model
          self.n_subjects   = 20
          self.n_conditions = 4   # FF, EF, FT, ET
          self.n_channels   = 56  #56
          self.n_trial      = 36  # For Imaginary it should be 22
          self.file_path    = './data/'  # Path to save the h5 files
          self.n_classes    = 1

          self.preprocess_normalize = False
          self.preprocess_scale = False
          self.device       = 'cuda:0'
          self.apply_valid_set = 'double'  # This defines to how many pieces we want to split the data ('all', 'double', 'single')
          self.realVSFake   = True
          self.data_path    = 'Img'    # Img vs Obs
          self.downsample   = False
          self.transform    = ""    # Specifies the transformation to be applied to the input (e.g., stockwell, stft)

          # Stockwell
          self.channels_list = [15,17,18, 20,43,44,46,48,49]
#          self.channels_list = [i for i in range(0,self.n_channels)]


          # Majority Voting Properties
          self.maj_vote    = False
          self.threshold   = 0.5

          # Gaussian Transform
          self.apply_gauss = False
          self.mapping_size = 3

          # Each segement
          if self.downsample:
             self.n_timepoints = 500
             self.window_len   = 480
             self.window_inc   = 2
          else:
             self.n_timepoints = 2000
             self.window_len  = 1900
             self.window_inc  = 10
          self.n_windows   = ((self.n_timepoints-self.window_len)// self.window_inc+1)

          # Stockwell Hyper-parameters
          self.fmax     = 30
          self.fmin     = 0
          self.sig_time = 2

          # STFT Hyper-parameters
          self.stft        = False
          self.fs          = 1000
          self.nperseg     = 500
          self.noverlap    = 485
          self.freq_cut    = 101
#          self.nTimeBins   = 2*(self.window_len - self.nperseg) // (self.nperseg - self.noverlap) + 1
          self.nTimeBins   = 108
    
          # Bad Subjects: [0,5,10,11,18]
          self.train_subjects  = []
          self.all_subjects    = np.array([1,2,3,4,6,7,8,9,12,13,14,15,16,17,19])
#          self.all_subjects    = np.array([1,2,3,4,6,7,8])
          self.valid_subjects  = []
          self.test_subjects   = []
          if self.realVSFake:
             self.conditions      = [0,1,2,3]
          else:
             self.conditions      = [2,3]

          if model == 'capsnet':
            # CNN (cnn)
            self.cnn_in_channels  = len(self.channels_list)
            if self.apply_gauss:
               self.cnn_in_channels  = 2*self.mapping_size
            self.cnn_out_channels = 32
            self.cnn_kernel_size  = 5

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels  = self.cnn_out_channels
            self.pc_out_channels = 32
            self.pc_kernel_size  = 9
            self.pc_num_routes   = 32 * 4 * 4

            # Digit Capsule (dc)
            self.dc_num_capsules = 1
            self.dc_num_routes   = self.pc_num_routes
            self.dc_in_channels  = 48
            self.dc_out_channels = 1

            # Decoder
            self.input_width  = self.window_len
            self.input_height = self.n_channels

          if 'lstm' in model or 'combined' in model:
            self.hidden_dim = 64
            self.layer_dim  = 2

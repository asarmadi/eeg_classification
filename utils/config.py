import numpy as np

class Config:
      def __init__(self, model='cnn'):
          self.n_subjects   = 20
          self.n_conditions = 4   # FF, EF, FT, ET
          self.n_channels   = 56
          self.n_timepoints = 2000
          self.n_trial      = 36  # For Imaginary 
          
          self.file_path    = './data/'  # Path to save the h5 files
          
          self.window_len  = 1000
          self.window_inc  = 20
          self.n_windows   = ((self.n_timepoints-self.window_len)// self.window_inc+1)
    
          # Bad Subjects: [0,5,10,11,18]
          self.train_subjects = [1,2,3,4,6,8,9,12,13,14,15,16,19]
          # train_subjects  = [1,2,3]
          self.valid_subjects  = [7]
          self.test_subjects   = [17]

          if model == 'caspnet':
            # CNN (cnn)
            self.cnn_in_channels = self.n_channels
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28               

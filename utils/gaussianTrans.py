import torch
import numpy as np

def GaussianFourierFeatureTransform(x, config):
        """
        An implementation of Gaussian Fourier feature mapping.

        "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
           https://arxiv.org/abs/2006.10739
           https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        Given an input of size [batches, num_input_channels, width, height],
         returns a tensor of size [batches, mapping_size*2, width, height].
        """

        scale = 10

        _num_input_channels = 1
        _mapping_size = config.mapping_size
        _B = torch.randn((_num_input_channels, _mapping_size)) * scale

        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == _num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(_num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ _B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, _mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

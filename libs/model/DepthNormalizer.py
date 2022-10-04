import torch
import torch.nn as nn

class DepthNormalizer(nn.Module):
    def __init__(self):
        super(DepthNormalizer, self).__init__()

    def forward(self, z, calibs=None, index_feat=None):
        '''
        Normalize z_feature
        :param z_feat: [B, 1, N] depth value for z in the image coordinate system
        :return:
        '''
        rng = 5.0
        sz = 256.0
        
        # z_feat = (z - (rng*0.5))*sz/rng
        z_feat = z*sz/rng

        return z_feat


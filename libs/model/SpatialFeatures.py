import torch
import numpy as np


class SpatialFeatures(torch.nn.Module):
    """
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, l):
        y = x[0]
        feats = []
        for i in range(l.shape[1]):
            cur_feat = y - l[:,i].reshape((-1,1))
            feats.append(cur_feat)
        feats = torch.cat(feats, 0).unsqueeze(0)
        return feats

        
from libs.data import TrainData, TestData
from libs.model import GIF_LGN
from libs.trainer import GIFTrainer

import torch
import os
import numpy as np
import cv2
from libs.assets import da_theta, CALLIBS
from libs.geometry import orthogonal

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda:1')

class OPTCLASS:
    def __init__(self):
        self.num_stack = 4
        self.norm = 'group'
        self.norm_color = 'instance'
        self.num_hourglass = 2
        self.hourglass_dim = 256
        self.mlp_dim_gif = [256+1+3, 512, 256, 128, 1]
        self.num_layers = 3
        self.gif_position = True # 3
        self.fourier = False # 36
        self.spatial = False # 171
        self.in_ch = 9
        self.garments = ['shirt', 'pant', 'coat', 'skirt', 'dress']

opt = OPTCLASS()
network = GIF_LGN(opt, device)

network.ind_completer['shirt'].load_state_dict(torch.load('checkpoints/ind/shirt_ind.pth', map_location='cuda:0'))
network.ind_completer['pant'].load_state_dict(torch.load('checkpoints/ind/pant_ind.pth', map_location='cuda:0'))
network.ind_completer['coat'].load_state_dict(torch.load('checkpoints/ind/coat_ind.pth', map_location='cuda:0'))
network.ind_completer['skirt'].load_state_dict(torch.load('checkpoints/ind/skirt_ind.pth', map_location='cuda:0'))
network.ind_completer['dress'].load_state_dict(torch.load('checkpoints/ind/dress_ind.pth', map_location='cuda:0'))


test_dataset = TestData(device)
print (len(test_dataset), 'test samples')

# fi, img, pgn, ref, calibs, glist = test_dataset[0]
# for g in opt.garments:
#     network.filter(pgn, ref[g], g)
#     mask = network.mask
#     save_img = mask.detach().cpu().numpy()[0]
#     save_img = np.transpose(save_img, (1,2,0))
#     save_img = (save_img*255).astype(np.uint8)
#     cv2.imwrite(f'ind_{g}.png', save_img)

# exit()

agent = GIFTrainer(network, device)
agent.load('checkpoints/network/gif_network_ckpt.pt')

for data in test_dataset:
    fi, img, pgn, ref, calibs, glist = data
    agent.eval(fi+'_ind', glist, pgn, ref, calibs)



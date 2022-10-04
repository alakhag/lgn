from libs.data import TrainData, TestData
from libs.model import LGN
from libs.trainer import Trainer

import torch
import os
import numpy as np
import cv2
from libs.assets import da_theta, CALLIBS
from libs.geometry import orthogonal

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda:0')

class OPTCLASS:
    def __init__(self):
        self.num_stack = 4
        self.norm = 'group'
        self.norm_color = 'instance'
        self.num_hourglass = 2
        self.hourglass_dim = 256
        self.mlp_dim = [256+1, 512, 256, 128, 1]
        self.num_layers = 3
        self.sdf_position = False # 3
        self.fourier = False # 36
        self.spatial = False # 171
        self.in_ch = 9
        self.garments = ['shirt', 'pant', 'coat', 'skirt', 'dress']

opt = OPTCLASS()
network = LGN(opt, device)

network.normalF.load_state_dict(torch.load("checkpoints/pix2pix/netF.pt", map_location='cuda:0'))
network.normalB.load_state_dict(torch.load("checkpoints/pix2pix/netB.pt", map_location='cuda:0'))
network.mask_completer['shirt'].load_state_dict(torch.load('checkpoints/mask_complete/shirt_complete.pth', map_location='cuda:0'))
network.mask_completer['pant'].load_state_dict(torch.load('checkpoints/mask_complete/pant_complete.pth', map_location='cuda:0'))
network.mask_completer['coat'].load_state_dict(torch.load('checkpoints/mask_complete/coat_complete.pth', map_location='cuda:0'))
network.mask_completer['skirt'].load_state_dict(torch.load('checkpoints/mask_complete/skirt_complete.pth', map_location='cuda:0'))
network.mask_completer['dress'].load_state_dict(torch.load('checkpoints/mask_complete/dress_complete.pth', map_location='cuda:0'))


test_dataset = TestData(device)
print (len(test_dataset), 'test samples')

# fi, img, pgn, ref, calibs, glist = test_dataset[0]
# for g in opt.garments:
#     network.filter(img, pgn, ref[g], g)
#     mask = network.mask
#     save_img = mask.detach().cpu().numpy()[0]
#     save_img = np.transpose(save_img, (1,2,0))
#     save_img = (save_img*255).astype(np.uint8)
#     cv2.imwrite(f'complete_{g}.png', save_img)

# exit()
agent = Trainer(network, device)
agent.load('checkpoints/network/network_ckpt.pt')

for data in test_dataset:
    fi, img, pgn, ref, calibs, glist = data
    print (fi)
    agent.eval(fi+'_rec', glist, img, pgn, ref, calibs)



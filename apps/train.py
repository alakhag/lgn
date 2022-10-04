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
        self.mlp_dim = [256+1, 512, 256, 128, 1]
        self.num_layers = 3
        self.position = False # 3
        self.fourier = False # 36
        self.spatial = False # 171
        self.in_ch = 9
        self.garments = ['shirt', 'pant', 'coat', 'skirt', 'dress']

opt = OPTCLASS()
print ('define network')
network = LGN(opt, device)

print ('load normal subnets')
network.normalF.load_state_dict(torch.load("checkpoints/pix2pix/netF.pt"))
network.normalB.load_state_dict(torch.load("checkpoints/pix2pix/netB.pt"))
print ('load mask generator')
print ('\t', 'shirt')
network.mask_completer['shirt'].load_state_dict(torch.load('checkpoints/mask_complete/shirt_complete.pth'))
print ('\t', 'pant')
network.mask_completer['pant'].load_state_dict(torch.load('checkpoints/mask_complete/pant_complete.pth'))
print ('\t', 'coat')
network.mask_completer['coat'].load_state_dict(torch.load('checkpoints/mask_complete/coat_complete.pth'))
print ('\t', 'skirt')
network.mask_completer['skirt'].load_state_dict(torch.load('checkpoints/mask_complete/skirt_complete.pth'))
print ('\t', 'dress')
network.mask_completer['dress'].load_state_dict(torch.load('checkpoints/mask_complete/dress_complete.pth'))


dataset = TrainData(device)
print (len(dataset), 'number of data...')
agent = Trainer(network, device)
agent.load('checkpoints/network_ckpt.pt')


for epoch in range(1000):
    # Epoch {epoch+1}: 
    print (f'Epoch {epoch+1}:')
    for i,data in enumerate(dataset):
        print (f'{i+1}/{len(dataset)}')
        agent.train(data)

        if ((i+1)%1000) == 0:
            torch.save(
            {'epoch':epoch,
            'encoder':network.encoder.state_dict(),
            'decoder':{x:network.decoders[x].state_dict() for x in opt.garments},
            'decoder_body': network.decoders['body'].state_dict()
            },
            'network_ckpt.pt')

    
    torch.save(
    {'epoch':epoch,
    'encoder':network.encoder.state_dict(),
    'decoder':{x:network.decoders[x].state_dict() for x in opt.garments},
    'decoder_body': network.decoders['body'].state_dict()
    },
    'network_ckpt.pt')


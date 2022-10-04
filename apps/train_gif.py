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
        self.mlp_dim = [256+1+3, 512, 256, 128, 1]
        self.num_layers = 3
        self.position = True # 3
        self.fourier = False # 36
        self.spatial = False # 171
        self.in_ch = 9
        self.garments = ['shirt', 'pant', 'coat', 'skirt', 'dress']

opt = OPTCLASS()
network = GIF_LGN(opt, device)

network.ind_completer['shirt'].load_state_dict(torch.load('checkpoints/ind/shirt_ind.pth'))
network.ind_completer['pant'].load_state_dict(torch.load('checkpoints/ind/pant_ind.pth'))
network.ind_completer['coat'].load_state_dict(torch.load('checkpoints/ind/coat_ind.pth'))
network.ind_completer['skirt'].load_state_dict(torch.load('checkpoints/ind/skirt_ind.pth'))
network.ind_completer['dress'].load_state_dict(torch.load('checkpoints/ind/dress_ind.pth'))
# print ('load mask generator')
# print ('\t', 'shirt')
# network.mask_completer['shirt'].load_state_dict(torch.load('checkpoints/mask_complete/shirt_complete.pth'))
# print ('\t', 'pant')
# network.mask_completer['pant'].load_state_dict(torch.load('checkpoints/mask_complete/pant_complete.pth'))
# print ('\t', 'coat')
# network.mask_completer['coat'].load_state_dict(torch.load('checkpoints/mask_complete/coat_complete.pth'))
# print ('\t', 'skirt')
# network.mask_completer['skirt'].load_state_dict(torch.load('checkpoints/mask_complete/skirt_complete.pth'))
# print ('\t', 'dress')
# network.mask_completer['dress'].load_state_dict(torch.load('checkpoints/mask_complete/dress_complete.pth'))

dataset = TrainData(device)
# test_dataset = TestData(device)
print (len(dataset), 'number of data...')

agent = GIFTrainer(network, device)
# agent.load('gif_network_ckpt.pt')

for epoch in range(1000):
    for i,data in enumerate(dataset):
        print (f'Epoch {epoch+1}: {i+1}/{len(dataset)}')
        # print (i)
        agent.train(data)

        if ((i+1)%1000) == 0:
            torch.save(
            {'epoch':epoch,
            # 'encoder':{x:network.encoders[x].state_dict() for x in opt.garments},
            'encoder':network.encoder.state_dict(),
            'decoder':{x:network.decoders[x].state_dict() for x in opt.garments},
            # 'decoder_body': network.decoders['body'].state_dict()
            },
            'gif_network_ckpt.pt')

            # for data in test_dataset:
            #     img, pgn, ref, calibs = data
            #     agent.eval('sample/gif/rec', ['shirt', 'pant', 'coat'], img, pgn, ref, calibs)
            #     test_img = (np.transpose(img[0].detach().cpu().numpy(), (1, 2, 0))*0.5 + 0.5)[:, :, ::-1] * 255.0
            #     cv2.imwrite('sample/sdf/test.png', test_img)
            

    torch.save(
    {'epoch':epoch,
    # 'encoder':{x:network.encoders[x].state_dict() for x in opt.garments},
    'encoder':network.encoder.state_dict(),
    'decoder':{x:network.decoders[x].state_dict() for x in opt.garments},
    # 'decoder_body': network.decoders['body'].state_dict()
    },
    'gif_network_ckpt.pt')

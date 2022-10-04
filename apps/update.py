import trimesh
from libs.data import TestData
from libs.model import LGN, GIF_LGN
from libs.trainer import Trainer, GIFTrainer

import torch
import os
import numpy as np
import cv2
from libs.assets import da_theta, CALLIBS
from libs.geometry import orthogonal
from skimage.measure import marching_cubes
from libs.assets import COLORS
from libs.save_util import save_obj_mesh_with_color
import pickle as pkl

if torch.cuda.is_available():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device1 = torch.device('cuda:0')
    device0 = torch.device('cuda:1')

class OPTCLASS:
    def __init__(self):
        self.num_stack = 4
        self.norm = 'group'
        self.norm_color = 'instance'
        self.num_hourglass = 2
        self.hourglass_dim = 256
        self.mlp_dim = [256+1, 512, 256, 128, 1]
        self.mlp_dim_gif = [256+1+3, 512, 256, 128, 1]
        self.num_layers = 3
        self.sdf_position = False # 3
        self.gif_position = True # 3
        self.fourier = False # 36
        self.spatial = False # 171
        self.in_ch = 9
        self.garments = ['shirt', 'pant', 'coat', 'skirt', 'dress']



eps = 5e-3
def update_sdf(SDFs, GIFs):
    for x in range(256):
        for y in range(256):
            for z in range(256):
                for gj in range(1,len(SDFs)):
                    for gi in range(gj):
                        if (GIFs[gi][x,y,z]>0.5 and GIFs[gj][x,y,z]>0.5) and SDFs[gj][x,y,z] > SDFs[gi][x,y,z] - eps:
                            SDFs[gj][x,y,z] = SDFs[gi][x,y,z] - eps
    
    return SDFs

def calc_full(SDFs, GIFs):
    sdf_full = np.ones((256,256,256))
    for x in range(256):
        for y in range(256):
            for z in range(256):
                val = SDFs[0][x,y,z]
                for gj in range(1,len(SDFs)):
                    if GIFs[gj][x,y,z]>0.5:
                        val = min(val, SDFs[gj][x,y,z])
                sdf_full[x,y,z] = val
    return sdf_full

origin = [-1, -1.34, -1]
spacing = 2.0/256.0
resX = 256
resY = 256
resZ = 256


opt = OPTCLASS()
network = LGN(opt, device0)
network.normalF.load_state_dict(torch.load("checkpoints/pix2pix/netF.pt", map_location=device0))
network.normalB.load_state_dict(torch.load("checkpoints/pix2pix/netB.pt", map_location=device0))
network.mask_completer['shirt'].load_state_dict(torch.load('checkpoints/mask_complete/shirt_complete.pth', map_location=device0))
network.mask_completer['pant'].load_state_dict(torch.load('checkpoints/mask_complete/pant_complete.pth', map_location=device0))
network.mask_completer['coat'].load_state_dict(torch.load('checkpoints/mask_complete/coat_complete.pth', map_location=device0))
network.mask_completer['skirt'].load_state_dict(torch.load('checkpoints/mask_complete/skirt_complete.pth', map_location=device0))
network.mask_completer['dress'].load_state_dict(torch.load('checkpoints/mask_complete/dress_complete.pth', map_location=device0))


network2 = GIF_LGN(opt, device1)
network2.ind_completer['shirt'].load_state_dict(torch.load('checkpoints/ind/shirt_ind.pth', map_location=device1))
network2.ind_completer['pant'].load_state_dict(torch.load('checkpoints/ind/pant_ind.pth', map_location=device1))
network2.ind_completer['coat'].load_state_dict(torch.load('checkpoints/ind/coat_ind.pth', map_location=device1))
network2.ind_completer['skirt'].load_state_dict(torch.load('checkpoints/ind/skirt_ind.pth', map_location=device1))
network2.ind_completer['dress'].load_state_dict(torch.load('checkpoints/ind/dress_ind.pth', map_location=device1))

agent = Trainer(network, device0)
agent.load('checkpoints/network/network_ckpt.pt')

agent2 = GIFTrainer(network2, device1)
agent2.load('checkpoints/network/gif_network_ckpt.pt')



test_dataset = TestData(device0)
print (len(test_dataset), 'test samples')
# data = test_dataset[0]
# fi, img, pgn, ref, calibs, glist = data
# test_dataset = [test_dataset[0]]
for data in test_dataset:
    fi, img, pgn, ref, calibs, glist = data
    print (fi)
    print ('\tbody')
    SDFs = [agent.get_sdf(img, pgn, ref, glist[0], calibs)]
    for g in glist[1:]:
        print ('\t',g)
        SDFs.append(agent.get_sdf(img, pgn, ref, g, calibs))

    pkl.dump(SDFs, open(fi+'sdfs.pkl', 'wb'))

    pgn = pgn.to(device1)
    for g in ref:
        ref[g] = ref[g].to(device1)
    calibs = calibs.to(device1)

    GIFs = [np.ones_like(SDFs[0])]
    # GIFs = []
    for g in glist[1:]:
        print ('\t',g)
        GIFs.append(agent2.get_sdf(pgn, ref, g, calibs))

    pkl.dump(GIFs, open(fi+'gifs.pkl', 'wb'))
    # GIFs = pkl.load(open('gifs.pkl', 'rb'))


    SDFs = update_sdf(SDFs, GIFs)
    SDF_full = calc_full(SDFs, GIFs)
    for i, sdf in enumerate(SDFs):
        print ('\t',i, sdf.min(), sdf.max())
        g = glist[i]
        verts, faces, _, _ = marching_cubes(sdf, 0)

        verts[:, 0] *= spacing
        verts[:, 1] *= spacing
        verts[:, 2] *= spacing
        verts = verts + np.array(origin).reshape((1, -1))

        color = np.zeros(verts.shape)
        color_ref = COLORS[g]
        color[:, 0] = color_ref[0]
        color[:, 1] = color_ref[1]
        color[:, 2] = color_ref[2]
        faces = faces[:, ::-1]

        save_obj_mesh_with_color(fi+f'_{g}.obj', verts, faces, color)

    verts, faces, _, _ = marching_cubes(SDF_full, 0)
    verts[:, 0] *= spacing
    verts[:, 1] *= spacing
    verts[:, 2] *= spacing
    verts = verts + np.array(origin).reshape((1, -1))

    faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    nml = mesh.vertex_normals
    color = nml*0.5+0.5
    # color = np.zeros(verts.shape)

    save_obj_mesh_with_color(fi+'_full_no_update'+'.obj', mesh.vertices, mesh.faces, color)

    for i, sdf in enumerate(GIFs):
        if i==0:
            continue
        sdf = sdf[:256, :256, :256]
        print ('\t',i, sdf.min(), sdf.max())
        g = glist[i]
        verts, faces, _, _ = marching_cubes(sdf, 0.5)

        verts[:, 0] *= spacing
        verts[:, 1] *= spacing
        verts[:, 2] *= spacing
        verts = verts + np.array(origin).reshape((1, -1))

        color = np.zeros(verts.shape)
        color_ref = [0.75, 0.75, 0.75]
        color[:, 0] = color_ref[0]
        color[:, 1] = color_ref[1]
        color[:, 2] = color_ref[2]
        # faces = faces[:, ::-1]

        save_obj_mesh_with_color(fi+'_ind'+f'_{g}.obj', verts, faces, color)


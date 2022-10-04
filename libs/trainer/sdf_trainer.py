import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from tqdm import tqdm
import random
from random import randrange
import math
from pylab import *
from libs.sdf import *
from skimage.measure import marching_cubes
from libs.save_util import *
from libs.model.loss import IGRLoss
from ..net_util import getBack
from torch import autograd
from libs.assets import COLORS
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, net, device):
        self.net = net
        self.device = device

        self.optimizers = {}
        for garm in ['body', 'shirt', 'pant', 'coat', 'skirt', 'dress']:
            paramlist = self.net.decoders[garm].parameters()
            paramlist = list(paramlist) + list(self.net.encoder.parameters())
            self.optimizers[garm] = torch.optim.Adam(paramlist, 1e-4)

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.igr = IGRLoss()

        self.origin = [-1, -1.34, -1]
        self.spacing = 2.0/256.0
        self.resX = 256
        self.resY = 256
        self.resZ = 256

        self.cube_verts, _ = create_grid(self.resX+1, self.resY+1, self.resZ+1, b_min=np.zeros(3),
                                         b_max=np.array([self.resX+1, self.resY+1, self.resZ+1]))

        self.eval_layer = -1
        self.test_calibs = None

        self.thresh = 0.0
        self.level = 0.0

    def train(self, data):
        img, pgn, ref, label, gif_label, calibs = data
        self.net.train()
        for garm in label:
            print (garm, end=' ')
            # with torch.no_grad():
            self.net.filter(img, pgn, ref[garm], garm)
            sdf_pred, gradient = self.net.query(label[garm]['P'], calibs, garm)
            loss = self.l1(sdf_pred, label[garm]['sdf'])
            self.optimizers[garm].zero_grad()
            loss.backward()
            self.optimizers[garm].step()
            print ("L1: {:.6f}".format(loss.item()), end=' ')

            # with torch.no_grad():
            self.net.filter(img, pgn, ref[garm], garm)
            sdf_surface, normal_surface = self.net.query(label[garm]['surface_pts'], calibs, garm)
            sdf_pred, gradient = self.net.query(label[garm]['P'], calibs, garm)
            loss = self.igr(sdf_surface, normal_surface, label[garm]['nmls'], sdf_pred, gradient)
            self.optimizers[garm].zero_grad()
            loss.backward()
            self.optimizers[garm].step()
        print ()

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.encoder.load_state_dict(ckpt['encoder'])
        for garm in ['shirt', 'pant', 'coat', 'skirt', 'dress']:
            self.net.decoders[garm].load_state_dict(ckpt['decoder'][garm])
        self.net.decoders['body'].load_state_dict(ckpt['decoder_body'])

    def calc(self, points):
        final_preds = []

        with torch.no_grad():
            for i in range(0, points.shape[1], 50000):
                pts = points[:, i:min(points.shape[1], i+50000)]
                _points = []
                for pt in pts.transpose((1, 0)):
                    U, V, H = pt
                    u1, v1, h1 = int(U), int(V), int(H)
                    u1 = self.origin[0] + u1*self.spacing
                    v1 = self.origin[1] + v1*self.spacing
                    h1 = self.origin[2] + h1*self.spacing
                    loc = np.array([u1, v1, h1])
                    _points.append(loc)
                _points = np.array(_points)
                _points = _points.transpose((1, 0))
                _points = np.expand_dims(_points, axis=0)
                samples = torch.from_numpy(_points).to(device=self.device).float()
                preds = self.net(self.img, self.pgn, self.ref, self.garm, samples, self.test_calibs)
                final_preds = final_preds + preds[0][0].detach().cpu().tolist()

            final_preds = np.array(final_preds)
            return final_preds

    def eval(self, filename, garments, img, pgn, ref, calibs):
        self.net.eval()
        with torch.no_grad():
            self.test_calibs = calibs
            self.img = img
            self.pgn = pgn
            for garm in garments:
                print (garm)
                self.garm = garm
                self.ref = ref[garm]
                sdf = eval_grid_octree(self.cube_verts, self.calc)
                print (sdf.max(), sdf.min())
                verts, faces, _, _ = marching_cubes(sdf, self.level)
                verts[:, 0] *= self.spacing
                verts[:, 1] *= self.spacing
                verts[:, 2] *= self.spacing
                verts = verts + np.array(self.origin).reshape((1, -1))

                color = np.zeros(verts.shape)
                color_ref = COLORS[garm]
                color[:, 0] = color_ref[0]
                color[:, 1] = color_ref[1]
                color[:, 2] = color_ref[2]
                faces = faces[:, ::-1]

                save_obj_mesh_with_color(filename+f'_{garm}.obj', verts, faces, color)

    def get_sdf(self, img, pgn, ref, garm, calibs):
        self.net.eval()
        with torch.no_grad():
            self.test_calibs = calibs
            self.img = img
            self.pgn = pgn
            self.garm = garm
            self.ref = ref[garm]
            sdf = eval_grid_octree(self.cube_verts, self.calc)
            return sdf



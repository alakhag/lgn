import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.autograd as autograd

from .SurfaceClassifier import SurfaceClassifier
from .HGFilters import HGFilter
from .BasePIFuNet import BasePIFuNet
from .DepthNormalizer import DepthNormalizer
from .PositionalEncoding import PositionalEncoding
from .SpatialFeatures import SpatialFeatures
from .PointDeformer import PointDeformer
from ..net_util import init_net, get_norm_layer
from ..networks import define_G
import numpy as np
import cv2
import pickle as pkl
from ..assets import da_theta, CALLIBS



class LGN(BasePIFuNet):
    def __init__(self, opt, device, projection_mode='orthogonal'):
        super(LGN, self).__init__(projection_mode=projection_mode)
        self.name = 'LGN'
        self.opt = opt

        self.features = [
            opt.sdf_position,
            opt.fourier,
            opt.spatial
        ]

        self.decoders = {}
        self.mask_completer = {}

        self.normalizer = DepthNormalizer()

        self.num_layers = opt.num_layers
        self.code = PositionalEncoding()
        self.psi = SpatialFeatures()

        self.in_ch = opt.in_ch
        self.encoder = HGFilter(opt, self.in_ch, 'down').to(device)

        dim = self.opt.mlp_dim.copy()
        for garm in opt.garments:
            self.decoders[garm] = SurfaceClassifier(
                    filter_channels=dim,
                    num_views=1,
                    no_residual=False,
                    last_op=nn.Sigmoid()
                ).to(device)
            self.mask_completer[garm] = define_G(4, 1, 64, "global", 4, 9, 1, 3, "instance").to(device)
        
        self.decoders['body'] = SurfaceClassifier(
                    filter_channels=dim,
                    num_views=1,
                    no_residual=False,
                    last_op=nn.Sigmoid()
                ).to(device)

        self.normalF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance").to(device)
        self.normalB = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance").to(device)
        self.nmlF = None
        self.nmlB = None
        self.mask = None

        self.device = device

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        init_net(self)


    def get_mask(self, pgn, ref, garm):
        with torch.no_grad():
            if garm!='body':
                inp = torch.cat((pgn,ref), 1)
                self.mask = self.mask_completer[garm](inp)
            else:
                self.mask = ref

    def get_nmlF(self, img):
        with torch.no_grad():
            self.nmlF = self.normalF.forward(img).detach()

    def get_nmlB(self, img):
        with torch.no_grad():
            self.nmlB = self.normalB.forward(img).detach()

    def filter(self, img, pgn, ref, garm):
        nmls = []
        with torch.no_grad():
            self.get_mask(pgn, ref, garm)
            self.get_nmlF(img)
            self.nmlF = self.mask.expand_as(self.nmlF)*self.nmlF
            nmls.append(self.nmlF)
            self.get_nmlB(img)
            self.nmlB = self.mask.expand_as(self.nmlB)*self.nmlB
            nmls.append(self.nmlB)
            nmls = torch.cat(nmls, 1)
            seg_img = self.mask.expand_as(img)*img
            # if garm=='shirt':
            #     save_img = self.mask.detach().cpu().numpy()[0]
            #     save_img = np.transpose(save_img, (1,2,0))
            #     save_img = (save_img*255).astype(np.uint8)
            #     cv2.imwrite('shirt_mask.png', save_img)

            #     save_img = self.nmlF.detach().cpu().numpy()[0]
            #     save_img = np.transpose(save_img, (1,2,0))
            #     save_img = (save_img*255).astype(np.uint8)
            #     cv2.imwrite('shirt_nmlf.png', save_img)

            #     save_img = self.nmlB.detach().cpu().numpy()[0]
            #     save_img = np.transpose(save_img, (1,2,0))
            #     save_img = (save_img*255).astype(np.uint8)
            #     cv2.imwrite('shirt_nmlb.png', save_img)

            #     save_img = seg_img.detach().cpu().numpy()[0]
            #     save_img = np.transpose(save_img, (1,2,0))
            #     save_img = (save_img*255).astype(np.uint8)
            #     cv2.imwrite('shirt_seg.png', save_img)
            #     exit()
            images = torch.cat([seg_img, nmls], 1)
        self.im_feat_list = self.encoder(images)
        # self.im_feat_list = [self.mask.expand_as(feat)*feat for feat in self.im_feat_list]
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, calibs, garm):
        with torch.enable_grad():
            points.requires_grad_()
            xyz = self.projection(points.cpu(), calibs.cpu())
            xyz = xyz.to(self.device)
            xy = xyz[:, :2, :]
            xy[:, 1, :] = xy[:, 1, :]*-1
            z = xyz[:, 2:3, :]
            z_feat = self.normalizer(z)

            fourier_feats = None
            spatial_feats = None

            if self.features[1]:
                fourier_feats = points/2.5
                fourier_feats = self.code(fourier_feats).to(self.device)

            if self.features[2]:
                if self.landmarks is None:
                    print('Error: No landmarks given')
                    exit()
                spatial_feats = self.psi(
                    points.cpu(), self.landmarks.cpu()).to(self.device)

            in_img = (xy[:, 0] > -1.0) & (xy[:, 0] <
                                        1.0) & (xy[:, 1] > -1.0) & (xy[:, 1] < 1.0)

            im_feat = self.im_feat_list[-1]
            point_local_feat_list = [self.index(im_feat, xy), z_feat]
            for i in range(len(self.features)):
                if self.features[i]:
                    if i == 0:
                        point_local_feat_list.append(points)
                    if i == 1:
                        point_local_feat_list.append(fourier_feats)
                    if i == 2:
                        point_local_feat_list.append(spatial_feats)
            point_local_feat = torch.cat(point_local_feat_list, 1)
            sdf, phi = self.decoders[garm](point_local_feat)
            sdf = (-2*in_img[:, None].float()*sdf+1).squeeze(0)
            
            if self.training:
                normal = autograd.grad(
                    [sdf.sum()], [points],
                    create_graph=True, retain_graph=True, only_inputs=True)[0].to(self.device)
            
            # if on_surface:
            #     normal = F.normalize(normal, dim=1, eps=1e-6)

            if self.training:
                self.preds = (sdf, normal)
            else:
                self.preds = (sdf, 0)

        return self.preds

    def get_preds(self):
        res = self.preds
        return res

    def forward(self, img, pgn, ref, garm, points, calibs):
        self.filter(img, pgn, ref, garm)

        self.query(points, calibs, garm)

        res = self.get_preds()

        return res


class GIF_LGN(BasePIFuNet):
    def __init__(self, opt, device, projection_mode='orthogonal'):
        super(GIF_LGN, self).__init__(projection_mode=projection_mode)
        self.name = 'GIF_LGN'
        self.opt = opt

        self.features = [
            opt.gif_position,   # position
            opt.fourier,
            opt.spatial
        ]

        self.normalizer = DepthNormalizer()

        self.code = PositionalEncoding()
        self.psi = SpatialFeatures()

        self.decoders = {}
        self.ind_completer = {}
        self.mask_completer = {}

        self.encoder = HGFilter(opt, 2, 'down').to(device)
        # self.encoders = {}
        dim = self.opt.mlp_dim_gif.copy()
        for garm in opt.garments:
            # self.encoders[garm] = HGFilter(opt, 2, 'down').to(device)
            self.decoders[garm] = SurfaceClassifier(
                    filter_channels=dim,
                    num_views=1,
                    no_residual=False,
                    last_op=nn.Sigmoid()
                ).to(device)
            self.ind_completer[garm] = define_G(4, 1, 64, "global", 4, 9, 1, 3, "instance").to(device)
            # self.mask_completer[garm] = define_G(4, 1, 64, "global", 4, 9, 1, 3, "instance").to(device)

        self.device = device

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.ind_mask = None
        self.mask = None

        init_net(self)


    def get_mask(self, pgn, ref, garm):
        with torch.no_grad():
            inp = torch.cat((pgn,ref), 1)
            self.ind_mask = self.ind_completer[garm](inp)
            # self.mask = self.mask_completer[garm](inp)

    def filter(self, pgn, ref, garm):
        with torch.no_grad():
            self.get_mask(pgn, ref, garm)
            # save_img = self.mask.detach().cpu().numpy()[0]
            # save_img = np.transpose(save_img, (1,2,0))
            # save_img = (save_img*255).astype(np.uint8)
            # cv2.imwrite('mask.png', save_img)
            # exit()
            images = torch.cat([ref, self.ind_mask], 1)
        self.im_feat_list = self.encoder(images)
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, calibs, garm):
        with torch.enable_grad():
            points.requires_grad_()
            xyz = self.projection(points.cpu(), calibs.cpu())
            xyz = xyz.to(self.device)
            xy = xyz[:, :2, :]
            xy[:, 1, :] = xy[:, 1, :]*-1
            z = xyz[:, 2:3, :]
            z_feat = self.normalizer(z)

            fourier_feats = None
            spatial_feats = None

            if self.features[1]:
                fourier_feats = points/2.5
                fourier_feats = self.code(fourier_feats).to(self.device)

            if self.features[2]:
                if self.landmarks is None:
                    print('Error: No landmarks given')
                    exit()
                spatial_feats = self.psi(
                    points.cpu(), self.landmarks.cpu()).to(self.device)

            in_img = (xy[:, 0] > -1.0) & (xy[:, 0] <
                                          1.0) & (xy[:, 1] > -1.0) & (xy[:, 1] < 1.0)

            im_feat = self.im_feat_list[-1]
            point_local_feat_list = [self.index(im_feat, xy), z_feat]
            for i in range(len(self.features)):
                if self.features[i]:
                    if i == 0:
                        point_local_feat_list.append(points)
                    if i == 1:
                        point_local_feat_list.append(fourier_feats)
                    if i == 2:
                        point_local_feat_list.append(spatial_feats)
            point_local_feat = torch.cat(point_local_feat_list, 1)
            sdf, phi = self.decoders[garm](point_local_feat)
            # gif = (-2*in_img[:, None].float()*sdf+1).squeeze(0)
            gif = (in_img[:, None].float()*sdf).squeeze(0)

            # if self.training:
            #     normal = autograd.grad(
            #         [gif.sum()], [points],
            #         create_graph=True, retain_graph=True, only_inputs=True)[0].to(self.device)

            # if on_surface:
            #     normal = F.normalize(normal, dim=1, eps=1e-6)

            if self.training:
                # self.preds = (gif, normal)
                self.preds = (gif, 0)
            else:
                self.preds = (gif, 0)

        return self.preds

    def get_preds(self):
        res = self.preds
        return res

    def forward(self, pgn, ref, garm, points, calibs):
        self.filter(pgn, ref, garm)

        self.query(points, calibs, garm)

        res = self.get_preds()

        return res

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from ..assets import da_theta
import pickle as pkl
import numpy as np

class PointDeformer(nn.Module):
    def __init__(self):
        super(PointDeformer, self).__init__()
        with open('assets/model_m.pkl', 'rb') as f:
            params = pkl.load(f, encoding='latin1')

        self.beta = torch.zeros(10).type(torch.float64)
        self.theta = torch.from_numpy(da_theta).type(torch.float64)
        self.J_regressor = torch.from_numpy(
            np.array(params['J_regressor'].todense())
        ).type(torch.float64)
        self.v_template = torch.from_numpy(params['v_template']).type(torch.float64)
        self.shapedirs = torch.from_numpy(params['shapedirs']).type(torch.float64)
        self.kintree_table = params['kintree_table']
        self.da_results = self.get_joint_transforms()
        for i in range(24):
            self.da_results[i] = torch.inverse(self.da_results[i])
            # self.da_results[i] = torch.eye(4, dtype=torch.float64)

    def get_joint_transforms(self):
        id_to_col = {self.kintree_table[1, i]: i
                     for i in range(self.kintree_table.shape[1])}
        parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        v_shaped = torch.tensordot(self.shapedirs, self.beta, dims = ([2], [0])) + self.v_template
        R_cube_big = self.rodrigues(self.theta.view(-1, 1, 3))
        J = torch.matmul(self.J_regressor, v_shaped)

        results = []
        results.append(
            self.with_zeros(
                torch.cat((R_cube_big[0], torch.reshape(J[0, :], (3, 1))), dim=1))
        )
        for i in range(1, self.kintree_table.shape[1]):
            results.append(
                torch.matmul(
                    results[parent[i]],
                    self.with_zeros(
                        torch.cat(
                            (R_cube_big[i], torch.reshape(
                                J[i, :] - J[parent[i], :], (3, 1))),
                            dim=1
                        )
                    )
                )
            )

        stacked = torch.stack(results, dim=0)
        results = stacked - \
            self.pack(
                torch.matmul(
                    stacked,
                    torch.reshape(
                        torch.cat(
                            (J, torch.zeros((24, 1), dtype=torch.float64)), dim=1),
                        (24, 4, 1)
                    )
                )
            )
        results = results.type(torch.FloatTensor)
        return results

    @staticmethod
    def rodrigues(r):
        eps = r.clone().normal_(std=1e-8)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=torch.float64).to(r.device)
        m = torch.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
            -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) \
                    + torch.zeros((theta_dim, 3, 3), dtype=torch.float64)).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def with_zeros(x):
        ones = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64).to(x.device)
        ret = torch.cat((x, ones), dim=0)
        return ret

    @staticmethod
    def pack(x):
        zeros43 = torch.zeros((x.shape[0], 4, 3), dtype=torch.float64).to(x.device)
        ret = torch.cat((zeros43, x), dim=2)
        return ret

    def set_beta(self, beta):
        self.beta = beta

    def set_theta(self, theta):
        self.theta = theta

    def forward(self, points, weights):
        points = torch.t(points)
        pose_results = self.get_joint_transforms()
        final_results = []
        for i in range(24):
            pose_results[i] = torch.matmul(pose_results[i], self.da_results[i]).type(torch.FloatTensor)
        
        T = torch.tensordot(weights, pose_results, dims=([0], [0]))
        rest_shape_h = torch.cat(
            (points, torch.ones((points.shape[0], 1), dtype=torch.float64)), dim=1
            ).type(torch.FloatTensor)
        v = torch.matmul(T, torch.reshape(rest_shape_h, (-1, 4, 1))).type(torch.FloatTensor)
        v = torch.reshape(v, (-1, 4))[:, :3]
        v = torch.t(v).unsqueeze(0)
        return v

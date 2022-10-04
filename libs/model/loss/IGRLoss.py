import torch
import torch.nn as nn

class IGRLoss(nn.Module):
    def __init__(self):
        super(IGRLoss, self).__init__()
        self.lamda_ls = 1.0
        self.lamda_nml = 1.0
        self.lamda_reg = 1.0
        self.lamda_nz = 0.1

    def forward(self, sdf_on_surface, normals_pred, normals_gt, sdf_pred, sdf_gradient):
        error_ls = self.lamda_ls*sdf_on_surface.abs().mean()
        error_nml = self.lamda_nml*torch.norm((normals_pred - normals_gt).abs(), p=2, dim=1).mean()
        error_reg = self.lamda_reg*((torch.norm(sdf_gradient, p=2, dim=1) - 1)**2).mean()
        error_nz = self.lamda_nz*torch.exp(-100*torch.abs(sdf_pred)).mean()
        error = error_ls + error_nml + error_reg + error_nz
        # error = error_ls
        print (
            '\tS:', "{:.4f}".format(error_ls.item()), 
            '\tNml:', "{:.4f}".format(error_nml.item()), 
            '\tReg:', "{:.4f}".format(error_reg.item()), 
            '\tnz:', "{:.4f}".format(error_nz.item()) 
            # end='\t'
            )
        return error

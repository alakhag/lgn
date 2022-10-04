from asyncore import read
from operator import imod
import numpy as np
import os
import trimesh
from tqdm import tqdm
from libs.data import TestData
import torch
import pickle as pkl
from skimage.measure import marching_cubes
from libs.save_util import save_obj_mesh_with_color

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda:1')

def read_obj(meshfile):
    color = [0,0,0]
    verts = []
    faces = []
    faces_info = []
    mesh = open(meshfile)
    for line in mesh:
        if line[0] == 'v':
            contents = line.strip().split()
            v1 = float(contents[1])
            v2 = float(contents[2])
            v3 = float(contents[3])
            verts.append([v1, v2, v3])
            color[0] = float(contents[4])
            color[1] = float(contents[5])
            color[2] = float(contents[6])
        else:
            faces_info.append(line.strip())
            contents = line.strip().split()
            v1 = int(contents[1])-1
            v2 = int(contents[2])-1
            v3 = int(contents[3])-1
            faces.append([v1, v2, v3])

    verts = np.array(verts)
    return verts, faces, faces_info, color



def interp(v1, v2, s1, s2):
    if s1 == 0 and s2 == 0:
        return (v1+v2)
    return (s2*v1 + s1*v2)/(s1+s2)
    # return (s1*v1 + s2*v2)/(s1+s2)
    # return (v1+v2)/2


def trim(verts, faces, vals, mark):
    new_faces = []
    new_verts = []
    N = verts.shape[0]
    for (i1, i2, i3) in faces:
        if mark[i1] and mark[i2] and mark[i3]:
            continue
        elif mark[i1] and mark[i2]:
            new_faces.append([i1, i2, i3])
            verts[i1] = interp(verts[i1], verts[i3], vals[i1], vals[i3])
            verts[i2] = interp(verts[i2], verts[i3], vals[i2], vals[i3])
        elif mark[i1] and mark[i3]:
            new_faces.append([i1, i2, i3])
            verts[i1] = interp(verts[i1], verts[i2], vals[i1], vals[i2])
            verts[i3] = interp(verts[i2], verts[i3], vals[i2], vals[i3])
        elif mark[i2] and mark[i3]:
            new_faces.append([i1, i2, i3])
            verts[i3] = interp(verts[i1], verts[i3], vals[i1], vals[i3])
            verts[i2] = interp(verts[i1], verts[i2], vals[i1], vals[i2])
        elif mark[i1]:
            vi = interp(verts[i1], verts[i2], vals[i1], vals[i2])
            vj = interp(verts[i1], verts[i3], vals[i1], vals[i3])
            i = N
            j = N+1
            N += 2
            new_verts.append(vi)
            new_verts.append(vj)
            new_faces.append([i, i2, i3])
            new_faces.append([i3, j, i])
        elif mark[i2]:
            vi = interp(verts[i2], verts[i3], vals[i2], vals[i3])
            vj = interp(verts[i1], verts[i2], vals[i1], vals[i2])
            i = N
            j = N+1
            N += 2
            new_verts.append(vi)
            new_verts.append(vj)
            new_faces.append([i, i3, i1])
            new_faces.append([i1, j, i])
        elif mark[i3]:
            vi = interp(verts[i1], verts[i3], vals[i1], vals[i3])
            vj = interp(verts[i2], verts[i3], vals[i2], vals[i3])
            i = N
            j = N+1
            N += 2
            new_verts.append(vi)
            new_verts.append(vj)
            new_faces.append([i, i1, i2])
            new_faces.append([i2, j, i])
        else:
            new_faces.append([i1, i2, i3])

    verts = np.array(verts.tolist() + new_verts)
    return verts, new_faces


def get_sdf(samples, obj_file):
    scan = trimesh.load(obj_file)
    fs = open('garment_pts.obj', 'w')
    for [X, Y, Z] in samples:
        fs.write('v '+str(X)+' '+str(Y)+' '+str(Z)+' 1 1 1\n')
    fs.close()

    os.system('~/libigl/build/tutorial/Calc_Winding_num_bin ' +
              obj_file+' garment_pts.obj winding_num.txt;')
    label_occ = np.zeros(samples.shape[0])
    label_npt = np.zeros_like(samples)

    winding_file = open('winding_num.txt').readlines()
    winding_num = np.array([float(x.strip()) for x in winding_file])
    occupancy = (winding_num > 0.5)*-2 + 1
    winding = winding_num
    winding[winding < 0.01] = 0.01
    winding[winding > 0.99] = 0.99
    winding = 0.5-winding
    os.remove('winding_num.txt')
    os.remove("garment_pts.obj")

    if samples.shape[0] >= 1000:
        I = 0
        s_old = ''
        for n in tqdm(np.array_split(samples[:(samples.shape[0]//1000)*1000], (samples.shape[0]//1000))):
            (_npt, _sdf, _f) = scan.nearest.on_surface(n)
            label_occ[1000*I:1000*(I+1)] = _sdf
            label_npt[1000*I:1000*(I+1), :] = _npt
            I += 1
        if samples.shape[0] % 1000 != 0:
            n = samples[(samples.shape[0]//1000)*1000:]
            (_npt, _sdf, _f) = scan.nearest.on_surface(n)
            label_occ[1000*I:] = _sdf
            label_npt[1000*I:, :] = _npt
    else:
        (label_npt, label_occ, label_nf) = scan.nearest.on_surface(samples)

    for i in range(label_occ.shape[0]):
        label_occ[i] = label_occ[i]*occupancy[i]

    return label_occ


origin = [-1, -1.34, -1]
spacing = 2.0/256.0
resX = 256
resY = 256
resZ = 256

test_dataset = TestData(device)

for data in test_dataset:
    fi, img, pgn, ref, calibs, glist = data
    print (fi)
    g = 'shirt'
    if os.path.exists(fi+'_trim'+f'_{g}.obj'):
        continue
    if not os.path.exists(fi+'_ind'+f'_{g}.obj'):
        continue
    # gifs = pkl.load(open(fi+'gifs.pkl', 'rb'))
    for i,g in enumerate(glist[1:]):
        # sdf = gifs[i+1]
        # sdf = sdf[:256, :256, :256]
        # print ('\t',i+1, sdf.min(), sdf.max())
        # verts, faces, _, _ = marching_cubes(sdf, 0.5)

        # verts[:, 0] *= spacing
        # verts[:, 1] *= spacing
        # verts[:, 2] *= spacing
        # verts = verts + np.array(origin).reshape((1, -1))

        # color = np.zeros(verts.shape)
        # color_ref = [0.75, 0.75, 0.75]
        # color[:, 0] = color_ref[0]
        # color[:, 1] = color_ref[1]
        # color[:, 2] = color_ref[2]
        # # faces = faces[:, ::-1]

        # save_obj_mesh_with_color(fi+'_ind'+f'_{g}.obj', verts, faces, color)
        
        obj_file = fi+'_ind'+f'_{g}.obj'
        gif_mesh = trimesh.load(obj_file)
        verts, faces, faces_info, color = read_obj(fi+f'_{g}.obj')
        out_file = fi+'_trim'+f'_{g}.obj'

        sdf = get_sdf(verts, obj_file)
        vals = np.abs(sdf)
        mark = (sdf>=0)
        verts, faces = trim(verts, faces, vals, mark)
        f = open(out_file, 'w')
        for v in verts:
            f.write('v '+' '.join([str(x) for x in v])+'\n')

        for v in faces:
            f.write('f '+' '.join([str(x+1) for x in v])+'\n')

        f.close()



# g = 'pant'
# obj_file = f'sample/rec_real/0.png_ind_{g}.obj'
# gif_mesh = trimesh.load(obj_file)
# verts, faces, faces_info, color = read_obj(f'sample/rec_real/0.png_update_{g}.obj')
# out_file = f'sample/rec_real/0.png_trim_{g}.obj'

# sdf = get_sdf(verts, obj_file)
# vals = np.abs(sdf)
# mark = (sdf>=0)
# verts, faces = trim(verts, faces, vals, mark)
# f = open(out_file, 'w')
# for v in verts:
#     f.write('v '+' '.join([str(x) for x in v])+'\n')

# for v in faces:
#     f.write('f '+' '.join([str(x+1) for x in v])+'\n')

# f.close()


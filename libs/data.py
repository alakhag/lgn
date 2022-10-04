from sys import exec_prefix
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle as pkl
from .assets import *
import trimesh
import os
import random
from tqdm import tqdm

from .geometry import perspective, index, orthogonal
import cv2

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_mask(mask_file):
    mask = Image.open(mask_file)
    mask = mask.convert('L')
    mask = mask.point(lambda i: i>0 and 255)
    mask = transforms.ToTensor()(mask).float()
    return mask

def visualize(pts, sdf=None):
    import trimesh
    cols = np.zeros_like(pts)
    if sdf is not None:
        cols[sdf>0,1] = 255
        cols[sdf<0,0] = 255
    pc = trimesh.PointCloud(pts, colors=cols)
    pc.show()

class TrainData(Dataset):
    def __init__(self, device):
        self.device = device
        self.root = '../DATA/'
        self.render_folder = os.path.join(self.root, 'RENDERS')
        self.base = os.path.join(self.root, 'posed_individual_garments')
        self.lfolder = os.path.join(self.root, 'GARMENTS')
        self.label_base = os.path.join(self.root, 'LABELS', 'posed_individual_garments')
        self.label_lfolder = os.path.join(self.root, 'LABELS', 'GARMENTS')

        self.mask_vals = {
            'shirt': [0, 65, 145],
            'pant': [65, 0, 65],
            'coat': [65, 145, 0],
            'skirt': [65, 0, 145],
            'dress': [145, 65, 65]
        }

        self.img_files = []
        self.pgn_files = []
        self.label_files = []

        combo_types = ['shirt_pant_coat', 'pant_shirt_coat', 'skirt_shirt', 'dress']
        shirts = ['BMan0006-M4', 'CMan0010-M4']
        pants = ['BMan0101-M4', 'CMan0441-M4']
        coats = ['BMan0313-M4', 'CMan0009-M4']
        skirts = ['BWom0100-M4']
        dresses = ['mus0007m4o01a']
        for combo in combo_types:
            print (combo)
            combo_folder = os.path.join(self.render_folder, combo)
            files = os.listdir(combo_folder)
            for fi in tqdm(files):
                if 'pgn' in fi:
                    continue
                sub = fi.split('.')[0]
                label = {}
                pose, pant, shirt, coat, skirt, dress = sub.split('_')
                if not (pant in pants or pant==''):
                    continue
                if not (shirt in shirts or shirt==''):
                    continue
                if not (coat in coats or coat==''):
                    continue
                if not (skirt in skirts or skirt==''):
                    continue
                if not (dress in dresses or dress==''):
                    continue
                # if pose not in ['11']:
                #     continue
                # if pant not in ['', 'CMan0306-M4', 'BMan0312-M4']:
                #     continue
                # if shirt not in ['', 'CMan0311-M4', 'SMan0004-M4', 'SMan0506-M4']:
                #     continue
                # if skirt not in ['', 'CWom0302-M4']:
                #     continue
                # if coat not in ['CMan0201-M4']:
                #     continue
                if combo.split('_')[0] == 'pant':
                    layer0_file = os.path.join(self.label_base, f'pose_{pose}', 'smpl.pkl')
                    # self.check(layer0_file)
                    label['body'] = layer0_file
                    layer1_file = os.path.join(self.label_base, f'pose_{pose}', 'pant', pant+'_data.pkl')
                    self.check(layer1_file)
                    label['pant'] = layer1_file
                    layer2_file = os.path.join(self.label_lfolder, combo, f'pose_{pose}', pant, shirt, shirt+'_data.pkl')
                    self.check(layer2_file)
                    label['shirt'] = layer2_file
                elif combo.split('_')[0] == 'shirt':
                    layer0_file = os.path.join(self.label_base, f'pose_{pose}', 'smpl.pkl')
                    # self.check(layer0_file)
                    label['body'] = layer0_file
                    layer1_file = os.path.join(self.label_base, f'pose_{pose}', 'shirt', shirt+'_data.pkl')
                    self.check(layer1_file)
                    label['shirt'] = layer1_file
                    layer2_file = os.path.join(self.label_lfolder, combo, f'pose_{pose}', shirt, pant, pant+'_data.pkl')
                    self.check(layer2_file)
                    label['pant'] = layer2_file
                    if coat:
                        layer3_file = os.path.join(self.label_lfolder, combo, f'pose_{pose}', shirt, pant,'Coats', coat+'_data.pkl')
                        self.check(layer3_file)
                        label['coat'] = layer3_file
                elif combo.split('_')[0] == 'skirt':
                    layer0_file = os.path.join(self.label_base, f'pose_{pose}', 'smpl_female.pkl')
                    # self.check(layer0_file)
                    label['body'] = layer0_file
                    layer1_file = os.path.join(self.label_base, f'pose_{pose}', 'skirt', skirt+'_data.pkl')
                    self.check(layer1_file)
                    label['skirt'] = layer1_file
                    layer2_file = os.path.join(self.label_lfolder, combo, f'pose_{pose}', skirt, shirt, shirt+'_data.pkl')
                    self.check(layer2_file)
                    label['shirt'] = layer2_file
                else:
                    layer0_file = os.path.join(self.label_base, f'pose_{pose}', 'smpl_female.pkl')
                    # self.check(layer0_file)
                    label['body'] = layer0_file
                    layer1_file = os.path.join(self.label_base, f'pose_{pose}', 'dress', dress+'_data.pkl')
                    self.check(layer1_file)
                    label['dress'] = layer1_file
                        
                img_file = os.path.join(self.render_folder, combo, sub+'.png')
                self.img_files.append(img_file)
                pgn_file = os.path.join(self.render_folder, combo, sub+'.pgn.png')
                self.pgn_files.append(pgn_file)
                self.label_files.append(label)
        
        c = list(zip(self.img_files, self.pgn_files, self.label_files))
        random.shuffle(c)
        self.img_files, self.pgn_files, self.label_files = zip(*c)

    def __len__(self):
        return len(self.img_files)

    def check(self, fi):
        return
        # if not os.path.exists(fi):
        #     return False
        # with open(fi, 'rb') as f:
        #     data = pkl.load(f, encoding='latin1')
        #     if 'sdf' not in data:
        #         return False
        try:
            with open(fi, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                if 'ind_sdf' not in data:
                    print (fi)
        except:
            print (fi)

    def get_mask(self, pgn, garm):
        if garm == 'body':
            mask = pgn
            mask = mask.convert('L')
            mask = mask.point(lambda i: i>0 and 255)
            mask = transforms.ToTensor()(mask).float()
            return mask
        else:
            img = np.array(pgn)
            img = img[:,:,::-1]
            low = np.array([x-5 for x in self.mask_vals[garm]])
            high = np.array([x+5 for x in self.mask_vals[garm]])
            mask = cv2.inRange(img, low, high)
            mask = Image.fromarray(mask).convert('L')
            mask = transforms.ToTensor()(mask).float()
            return mask

    def __getitem__(self, idx):
        fi = self.img_files[idx]
        img = Image.open(fi).convert('RGB')
        img = to_tensor(img)
        img = img.float().unsqueeze(0).to(self.device)

        fi = self.pgn_files[idx]
        pgn = Image.open(fi).convert('RGB')

        ref = {}

        lfiles = self.label_files[idx]
        label = {}
        gif_label = {}
        for garm in lfiles:
            # try:
            with open(lfiles[garm], 'rb') as f:
                # print (lfiles[garm])
                data = pkl.load(f, encoding='latin1')
                if garm!='body':
                    ind_surface_pts = data['ind_surface_pts']
                    ind_nmls = data['ind_nmls']
                    ind_P = data['ind_P']
                    ind_sdf_ = data['ind_sdf']
                    # ind_sdf = data['ind_sdf']
                    ind_sdf = np.zeros_like(ind_sdf_)
                    ind_sdf[ind_sdf_<0] = 1
                    ind_sdf[ind_sdf_>0] = 0
                    # visualize(ind_P, ind_sdf)
                    ind_edges = data['ind_edges']
                    ind_sdf = torch.from_numpy(ind_sdf).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    ind_surface_pts = torch.from_numpy(ind_surface_pts.T).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    ind_edges = torch.from_numpy(ind_edges.T).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    ind_P = torch.from_numpy(ind_P.T).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    ind_nmls = torch.from_numpy(ind_nmls.T).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    gif_label[garm] = {'surface_pts': ind_surface_pts, 'nmls': ind_nmls, 'P': ind_P, 'sdf': ind_sdf, 'edge': ind_edges}
                surface_pts = data['surface_pts']
                nmls = data['nmls']
                P = data['P']
                # visualize(surface_pts)
                sdf = data['sdf']
                # visualize(P, sdf)
                # visualize(np.concatenate([surface_pts, P]))
                sdf = torch.from_numpy(sdf).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                surface_pts = torch.from_numpy(surface_pts.T).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                P = torch.from_numpy(P.T).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                nmls = torch.from_numpy(nmls.T).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                label[garm] = {'surface_pts': surface_pts, 'nmls': nmls, 'P': P, 'sdf': sdf}
                # exit()
            mask = self.get_mask(pgn, garm)
            ref[garm] = mask.unsqueeze(0).to(self.device)
            
            # except:
            #     continue

        pgn = to_tensor(pgn)
        pgn = pgn.float().unsqueeze(0).to(self.device)

        calibs = torch.Tensor(CALLIBS).to(self.device)

        return img, pgn, ref, label, gif_label, calibs

class TestData(Dataset):
    def __init__(self, device):
        self.device = device

        self.mask_vals = {
            'shirt': [0, 65, 145],
            'pant': [65, 0, 65],
            'coat': [65, 145, 0],
            'skirt': [65, 0, 145],
            'dress': [145, 65, 65]
        }

        # self.root = 'sample/rec_real'
        # self.img_files = []
        # self.pgn_files = []
        # for i in range(1):
        #     fi_png = os.path.join(self.root, '{}.png'.format(i))
        #     fi_jpg = os.path.join(self.root, '{}.jpg'.format(i))
        #     if os.path.exists(fi_png):
        #         self.img_files.append(fi_png)
        #         self.pgn_files.append(fi_png[:-4]+'_pgn.png')
        #     else:
        #         self.img_files.append(fi_jpg)
        #         self.pgn_files.append(fi_jpg[:-4]+'_pgn.jpg')
        # self.img_files = ['sample/rec_real/0.png',     'sample/sdf_real/test (copy)/0.png',     'sample/sdf_real/test/1.png']
        # # self.img_files = ['sample/sdf_real/test/1.png']
        # self.pgn_files = ['sample/rec_real/0_pgn.png',     'sample/sdf_real/test (copy)/0_pgn.png',     'sample/sdf_real/test/1_pgn.png']
        # self.garm_list = [
        #     ['body', 'coat'],
        #     ['body', 'coat'],
        #     ['body', 'coat']
        # ]


        # self.img_files = ['sample/synthetic/real/0.png']
        # self.pgn_files = ['sample/synthetic/real/0_pgn.png']
        # self.garm_list = [
        #     ['shirt']
        # ]
        # self.img_files = [
        #     # '/home/cgalab/AlakhAggarwal/lgn/sample/image_sizer/10069_6827.png', 
        # '/home/cgalab/AlakhAggarwal/lgn/sample/image_sizer/10110_9466.png']
        # self.pgn_files = [
        #     # '/home/cgalab/AlakhAggarwal/lgn/sample/image_sizer/10069_6827_pgn.png', 
        # '/home/cgalab/AlakhAggarwal/lgn/sample/image_sizer/10110_9466_pgn.png']
        # self.garm_list = [
        #     # ['body', 'pant', 'shirt'],
        #     ['body', 'skirt', 'shirt']
        # ]

        self.img_files = []
        self.pgn_files = []
        self.garm_list = []
        self.root = 'sample/data'
        datas = sorted(os.listdir(self.root))
        datas = [x for x in datas if x.endswith('.png') and '_pgn' not in x]
        I = 0
        for data in datas:
            data = data.split('.')[0]
        #     if os.path.exists(os.path.join(self.root, data+'.png_body.obj')):
        #         continue
        #     # if data[-3:]=='pgn':
        #     #     continue
        #     # I += 1
        #     # if I<=8:
        #     #     continue
            self.img_files.append(os.path.join(self.root, data+'.png'))
            self.pgn_files.append(os.path.join(self.root, data+'_pgn.png'))
            self.garm_list.append(['body', 'shirt', 'pant', 'coat'])
        #     if data == '10110_9466':
        #         self.garm_list.append(['body', 'skirt', 'shirt'])
        #     else:
        #         g_labels = ['body', 'pant', 'shirt']
        #         self.garm_list.append(g_labels)
        # self.garm_list = [
        #     # ['body', 'dress'],
        #     ['body', 'pant', 'shirt', 'coat'],
        #     ['body', 'pant', 'shirt', 'coat']
        # ]

            # data_folder = os.path.join(self.root, data)
            # label_file = os.path.join(self.root, 'labels', data+'.txt')
            # g_labels = open(label_file).readlines()
            # g_labels = [g_label.strip().split(',') for g_label in g_labels]
            # self.garm_list = self.garm_list + g_labels
            # for i in range(len(g_labels)):
            #     self.img_files.append(os.path.join(data_folder, '{}.png'.format(i)))
            #     self.pgn_files.append(os.path.join(data_folder, '{}_pgn.png'.format(i)))
        # self.img_files = ['sample/synthetic/dress/11.png']
        # self.pgn_files = ['sample/synthetic/dress/11_pgn.png']
        # self.garm_list = [['body', 'dress']]

    def __len__(self):
        return len(self.img_files)

    def get_mask(self, pgn, garm):
        if garm == 'body':
            mask = pgn
            mask = mask.convert('L')
            mask = mask.point(lambda i: i>0 and 255)
            mask = transforms.ToTensor()(mask).float()
            return mask
        else:
            img = np.array(pgn)
            img = img[:,:,::-1]
            low = np.array([x-5 for x in self.mask_vals[garm]])
            high = np.array([x+5 for x in self.mask_vals[garm]])
            mask = cv2.inRange(img, low, high)
            mask = Image.fromarray(mask).convert('L')
            mask = transforms.ToTensor()(mask).float()
            return mask

    def __getitem__(self, idx):
        fi = self.img_files[idx]
        img = Image.open(fi).convert('RGB')
        width, height = img.size
        left = (512-width)//2
        top = (512-height)//2
        img_new = Image.new(img.mode, (512,512), (0,0,0))
        img_new.paste(img, (left, top))
        img = img_new
        img = to_tensor(img)
        img = img.float().unsqueeze(0).to(self.device)

        fi = self.pgn_files[idx]
        pgn = Image.open(fi).convert('RGB')
        pgn_new = Image.new(pgn.mode, (512,512), (0,0,0))
        pgn_new.paste(pgn, (left, top))
        pgn = pgn_new
        glist = self.garm_list[idx]

        ref = {}

        for garm in glist:
            mask = self.get_mask(pgn, garm)
            ref[garm] = mask.unsqueeze(0).to(self.device)
        
        pgn = to_tensor(pgn)
        pgn = pgn.float().unsqueeze(0).to(self.device)

        calibs = torch.Tensor(CALLIBS).to(self.device)

        return self.img_files[idx], img, pgn, ref, calibs, glist


class ImageDataset(Dataset):
    def __init__(self, device):
        self.device = device
        self.root = '/home/cgalab/AlakhAggarwal/DATA'
        self.render_folder = os.path.join(self.root, 'RENDERS')
        self.base = os.path.join(self.root, 'posed_individual_garments')
        self.lfolder = os.path.join(self.root, 'GARMENTS')

        no_mask = np.zeros((512,512), dtype=np.uint8)
        no_mask = Image.fromarray(no_mask).convert('L')
        no_mask = no_mask.point(lambda i: i>0 and 255)
        self.no_mask = transforms.ToTensor()(no_mask).float()

        self.shirt_mask_val = [0, 65, 145]
        self.pant_mask_val = [65, 0, 65]
        self.coat_mask_val = [65, 145, 0]
        self.skirt_mask_val = [65, 0, 145]
        self.dress_mask_val = [145, 65, 65]

        self.img_files = []
        self.mask_label_files = []
        self.gif_label_files = []

        combo_types = ['shirt_pant_coat', 'pant_shirt_coat']
        for combo in combo_types:
            combo_folder = os.path.join(self.render_folder, combo)
            files = os.listdir(combo_folder)
            for fi in files:
                if 'pgn' in fi:
                    continue
                sub = fi.split('.')[0]
                mlabel = {'shirt': None, 'pant': None, 'coat': None, 'dress': None, 'skirt': None}
                hlabel = {'shirt': None, 'pant': None, 'coat': None, 'dress': None, 'skirt': None}
                pose, pant, shirt, coat, skirt, dress = sub.split('_')
                if combo.split('_')[0] == 'pant':
                    layer1_file = os.path.join(self.base, f'pose_{pose}', 'pant', pant+'_mask.png')
                    mlabel['pant'] = layer1_file
                    layer2_file = os.path.join(self.lfolder, combo, f'pose_{pose}', pant, shirt, shirt+'_mask.png')
                    mlabel['shirt'] = layer2_file
                    layer1_file = os.path.join(self.base, f'pose_{pose}', 'pant', pant+'_ind.png')
                    hlabel['pant'] = layer1_file
                    layer2_file = os.path.join(self.lfolder, combo, f'pose_{pose}', pant, shirt, shirt+'_ind.png')
                    hlabel['shirt'] = layer2_file
                    # continue
                elif combo.split('_')[0] == 'shirt':
                    layer1_file = os.path.join(self.base, f'pose_{pose}', 'shirt', shirt+'_mask.png')
                    mlabel['shirt'] = layer1_file
                    layer2_file = os.path.join(self.lfolder, combo, f'pose_{pose}', shirt, pant, pant+'_mask.png')
                    mlabel['pant'] = layer2_file
                    layer1_file = os.path.join(self.base, f'pose_{pose}', 'shirt', shirt+'_ind.png')
                    hlabel['shirt'] = layer1_file
                    layer2_file = os.path.join(self.lfolder, combo, f'pose_{pose}', shirt, pant, pant+'_ind.png')
                    hlabel['pant'] = layer2_file
                    if coat:
                        layer3_file = os.path.join(self.lfolder, combo, f'pose_{pose}', shirt, pant,'Coats', coat+'_mask.png')
                        mlabel['coat'] = layer3_file
                        layer3_file = os.path.join(self.lfolder, combo, f'pose_{pose}', shirt, pant,'Coats', coat+'_ind.png')
                        hlabel['coat'] = layer3_file
                    # else:
                    #     continue
                elif combo.split('_')[0] == 'skirt':
                    layer1_file = os.path.join(self.base, f'pose_{pose}', 'skirt', skirt+'_mask.png')
                    mlabel['skirt'] = layer1_file
                    layer2_file = os.path.join(self.lfolder, combo, f'pose_{pose}', skirt, shirt, shirt+'_mask.png')
                    mlabel['shirt'] = layer2_file
                    layer1_file = os.path.join(self.base, f'pose_{pose}', 'skirt', skirt+'_ind.png')
                    hlabel['skirt'] = layer1_file
                    layer2_file = os.path.join(self.lfolder, combo, f'pose_{pose}', skirt, shirt, shirt+'_ind.png')
                    hlabel['shirt'] = layer2_file
                    # continue
                else:
                    layer1_file = os.path.join(self.base, f'pose_{pose}', 'dress', dress+'_mask.png')
                    mlabel['dress'] = layer1_file
                    layer1_file = os.path.join(self.base, f'pose_{pose}', 'dress', dress+'_ind.png')
                    hlabel['dress'] = layer1_file
                    # continue
                
                self.mask_label_files.append(mlabel)
                self.gif_label_files.append(hlabel)
                pgn_file = os.path.join(self.render_folder, combo, sub+'.pgn.png')
                self.img_files.append(pgn_file)
        
        c = list(zip(self.img_files, self.mask_label_files, self.gif_label_files))
        random.shuffle(c)
        self.img_files, self.mask_label_files, self.gif_label_files = zip(*c)

    def __len__(self):
        return len(self.img_files)

    def get_shirt(self, img):
        shirt_low = np.array([x-5 for x in self.shirt_mask_val])
        shirt_high = np.array([x+5 for x in self.shirt_mask_val])
        mask = cv2.inRange(img, shirt_low, shirt_high)
        return mask

    def get_pant(self, img):
        pant_low = np.array([x-5 for x in self.pant_mask_val])
        pant_high = np.array([x+5 for x in self.pant_mask_val])
        mask = cv2.inRange(img, pant_low, pant_high)
        return mask

    def get_coat(self, img):
        coat_low = np.array([x-5 for x in self.coat_mask_val])
        coat_high = np.array([x+5 for x in self.coat_mask_val])
        mask = cv2.inRange(img, coat_low, coat_high)
        return mask

    def get_skirt(self, img):
        skirt_low = np.array([x-5 for x in self.skirt_mask_val])
        skirt_high = np.array([x+5 for x in self.skirt_mask_val])
        mask = cv2.inRange(img, skirt_low, skirt_high)
        return mask

    def get_dress(self, img):
        dress_low = np.array([x-5 for x in self.dress_mask_val])
        dress_high = np.array([x+5 for x in self.dress_mask_val])
        mask = cv2.inRange(img, dress_low, dress_high)
        return mask

    def get_seg(self, img):
        opencv_img = np.array(img)
        opencv_img = opencv_img[:,:,::-1]

        shirt_mask = self.get_shirt(opencv_img)
        shirt_mask = Image.fromarray(shirt_mask).convert('L')
        shirt_mask = transforms.ToTensor()(shirt_mask).float()

        pant_mask = self.get_pant(opencv_img)
        pant_mask = Image.fromarray(pant_mask).convert('L')
        pant_mask = transforms.ToTensor()(pant_mask).float()

        coat_mask = self.get_coat(opencv_img)
        coat_mask = Image.fromarray(coat_mask).convert('L')
        coat_mask = transforms.ToTensor()(coat_mask).float()

        skirt_mask = self.get_skirt(opencv_img)
        skirt_mask = Image.fromarray(skirt_mask).convert('L')
        skirt_mask = transforms.ToTensor()(skirt_mask).float()

        dress_mask = self.get_dress(opencv_img)
        dress_mask = Image.fromarray(dress_mask).convert('L')
        dress_mask = transforms.ToTensor()(dress_mask).float()

        return shirt_mask, pant_mask, coat_mask, skirt_mask, dress_mask


    def __getitem__(self, idx):
        fi = self.img_files[idx]
        img = Image.open(fi).convert('RGB')
        shirt_mask, pant_mask, coat_mask, skirt_mask, dress_mask = self.get_seg(img)
        img = to_tensor(img)
        img = img.float().unsqueeze(0).to(self.device)

        ref = {}
        ref['shirt'] = shirt_mask.float().unsqueeze(0).to(self.device)
        ref['pant'] = pant_mask.float().unsqueeze(0).to(self.device)
        ref['coat'] = coat_mask.float().unsqueeze(0).to(self.device)
        ref['skirt'] = skirt_mask.float().unsqueeze(0).to(self.device)
        ref['dress'] = dress_mask.float().unsqueeze(0).to(self.device)

        mfiles = self.mask_label_files[idx]
        hfiles = self.gif_label_files[idx]
        label = {}
        gif_label = {}
        for garm in mfiles:
            if mfiles[garm] is None:
                label[garm] = self.no_mask.float().unsqueeze(0).to(self.device)
                gif_label[garm] = self.no_mask.float().unsqueeze(0).to(self.device)
            else:
                fi = mfiles[garm]
                mask = get_mask(fi)
                label[garm] = mask.float().unsqueeze(0).to(self.device)
                fi = hfiles[garm]
                mask = get_mask(fi)
                gif_label[garm] = mask.float().unsqueeze(0).to(self.device)

        return img, ref, label, gif_label





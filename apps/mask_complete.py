from libs.data import *
from libs.networks import define_G
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
garments = [
    'shirt', 
    'pant', 
    'coat', 
    'skirt', 
    'dress'
    ]

dataset = ImageDataset(device)
print (len(dataset), 'number of data...')
test_imgs = []
inp_imgs = []
gt_imgs = []
root = '/home/cgalab/AlakhAggarwal/DATA'
render_folder = os.path.join(root, 'RENDERS')
files = [
    '/home/cgalab/AlakhAggarwal/DATA/RENDERS/shirt_pant_coat/0_CMan0306-M4_BMan0006-M4_CMan0201-M4__.pgn.png',
    '/home/cgalab/AlakhAggarwal/DATA/RENDERS/shirt_pant_coat/12_CMan0323-M4_BMan0006-M4_BMan0318-M4__.pgn.png',
    '/home/cgalab/AlakhAggarwal/DATA/RENDERS/shirt_pant_coat/11_CMan0010-M4_CMan0321-M4___.pgn.png',
    '/home/cgalab/AlakhAggarwal/DATA/RENDERS/shirt_pant_coat/12_CMan0324-M4_CMan0304-M4___.pgn.png',
    '/home/cgalab/AlakhAggarwal/DATA/RENDERS/skirt_shirt/16__SMan0505-M4__CWom0104-M4_.pgn.png',
    '/home/cgalab/AlakhAggarwal/DATA/RENDERS/dress/0_____CWom0319-M4.pgn.png'

]
for fi in files:
    sub = fi.split('.')[0]
    img = Image.open(fi).convert('RGB')
    mask = img
    mask = mask.convert('L')
    mask = mask.point(lambda i: i>0 and 255)
    shirt_mask, pant_mask, coat_mask, skirt_mask, dress_mask = dataset.get_seg(img)
    img = to_tensor(img)
    img = img.float().unsqueeze(0).to(device)
    test_imgs.append(img)

    ref = {}
    ref['shirt'] = shirt_mask.float().unsqueeze(0).to(device)
    ref['pant'] = pant_mask.float().unsqueeze(0).to(device)
    ref['coat'] = coat_mask.float().unsqueeze(0).to(device)
    ref['skirt'] = skirt_mask.float().unsqueeze(0).to(device)
    ref['dress'] = dress_mask.float().unsqueeze(0).to(device)
    gt_imgs.append(ref)

    ref['shirt'] = torch.cat((img, ref['shirt']), 1)
    ref['pant'] = torch.cat((img, ref['pant']), 1)
    ref['coat'] = torch.cat((img, ref['coat']), 1)
    ref['skirt'] = torch.cat((img, ref['skirt']), 1)
    ref['dress'] = torch.cat((img, ref['dress']), 1)
    inp_imgs.append(ref)


network_shirt = define_G(4, 1, 64, "global", 4, 9, 1, 3, "instance")
network_shirt = network_shirt.to(device)
optim_shirt = torch.optim.Adam(network_shirt.parameters(), 1e-4)

network_pant = define_G(4, 1, 64, "global", 4, 9, 1, 3, "instance")
network_pant = network_pant.to(device)
optim_pant = torch.optim.Adam(network_pant.parameters(), 1e-4)

network_coat = define_G(4, 1, 64, "global", 4, 9, 1, 3, "instance")
network_coat = network_coat.to(device)
optim_coat = torch.optim.Adam(network_coat.parameters(), 1e-4)

network_dress = define_G(4, 1, 64, "global", 4, 9, 1, 3, "instance")
network_dress = network_dress.to(device)
optim_dress = torch.optim.Adam(network_dress.parameters(), 1e-4)

network_skirt = define_G(4, 1, 64, "global", 4, 9, 1, 3, "instance")
network_skirt = network_skirt.to(device)
optim_skirt = torch.optim.Adam(network_skirt.parameters(), 1e-4)

network = {
    'shirt': network_shirt,
    'pant': network_pant,
    'coat': network_coat,
    'skirt': network_skirt,
    'dress': network_dress
}
optim = {
    'shirt': optim_shirt,
    'pant': optim_pant,
    'coat': optim_coat,
    'skirt': optim_skirt,
    'dress': optim_dress
}

# network['dress'].load_state_dict(torch.load('dress_complete.pth'))

for epoch in range(5):
    print (f'Epoch {epoch+1}')
    for i, data in enumerate(dataset):
        print ('\t', i, end = '\t')
        img = data[0]
        for garm in garments:
            ref = data[1][garm]
            inp = torch.cat((img, ref), 1)
            out = data[2][garm]
            pred = network[garm](inp)
            loss = torch.nn.L1Loss()(out, pred)
            optim[garm].zero_grad()
            loss.backward()
            optim[garm].step()
            print("{:.6f}".format(loss.item()), end=' ')
        print ()

    # for j, inp in enumerate(inp_imgs):
    #     out_folder = os.path.join('sample', 'mask_completion', str(j))
    #     if not os.path.exists(out_folder):
    #         os.makedirs(out_folder)
    #     # test_img = test_imgs[j]
    #     # gt = gt_imgs[j]
    #     # test_img = (np.transpose(test_img[0].detach().cpu().numpy(), (1, 2, 0))*0.5 + 0.5)[:, :, ::-1] * 255.0
    #     # save_path = os.path.join(out_folder, 'test.png')
    #     # cv2.imwrite(save_path, test_img)
    #     for garm in garments:
    #         print (garm)
    #         network[garm].eval()
    #         pred = network[garm](inp[garm])
    #         pred = (np.transpose(pred[0].detach().cpu().numpy(), (1, 2, 0)))[:, :, ::-1] * 255.0
    #         save_path = os.path.join(out_folder, garm+'.png')
    #         cv2.imwrite(save_path, pred)
    #         # gt_im = gt[garm]
    #         # gt_im = (np.transpose(gt_im[0].detach().cpu().numpy(), (1, 2, 0)))[:, :, ::-1] * 255.0
    #         # save_path = os.path.join(out_folder, 'gt_'+garm+'.png')
    #         # cv2.imwrite(save_path, gt_im)
            
    #         network[garm].train()

    #         torch.save(network[garm].state_dict(), garm+'_complete.pth')
    # # exit()

from AttUnetPP import UNet_Nested
import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from collections import OrderedDict
import torch.nn.functional as F
import time

Generator_StateDict = OrderedDict()
Color_StateDict = OrderedDict()

Generator = UNet_Nested()
Color = UNet_Nested()

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((1200, 1200)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

ckpt = torch.load('Ckpt/test-0565.pth', map_location='cuda:1')

Generator.load_state_dict(ckpt)
Generator.eval()
Generator.cuda(1)

valid_path = './Data/NTIRE2023_Valid_Hazy/'
test_path = './Data/NTIRE2023_Test_Hazy/'
test_pre_path = './TestResults/'

test_list = [test_path + i for i in os.listdir(test_path)] + [valid_path + i for i in os.listdir(valid_path)]
size = 800

start_time = time.time()

for test_img in test_list:
    raw_img = cv2.imread(test_img, cv2.IMREAD_UNCHANGED)
    w, h, _ = raw_img.shape
    if _ == 4:
        alpha = raw_img[..., 3:]
        raw_img = raw_img[..., :3]
    img = test_transform(raw_img)
    img = img.unsqueeze(0)
    print(img.shape)
    img_pad = F.pad(img, (int(size/4), int(size/4), int(size/4), int(size/4)), 'reflect')
    img_zero = torch.zeros(1, 3, w, h).cuda(3)
    with torch.no_grad():
        for i in range(int(w/size*2)):
            for j in range(int(h/size*2)):
                img_patch = img_pad[:, :, i*int(size/2):i*int(size/2)+size, j*int(size/2):j*int(size/2)+size].cuda(1)
                pred = Generator(img_patch)[:,:,int(size/4):int(size/4)+int(size/2), int(size/4):int(size/4)+int(size/2)]
                img_zero[:, :, i*int(size/2):i*int(size/2)+int(size/2), j*int(size/2):j*int(size/2)+int(size/2)] = pred
        pred = img_zero

        result = 255 * pred.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.double)
        if _ == 4:
            result = np.concatenate((result, alpha), axis=-1)
        if cv2.imwrite(test_pre_path+test_img[-6:], result):
            print(test_img)

from torch import nn
import random
import numpy as np
import torch
import torchvision
import torch
import numpy as np
import cv2
import torch.autograd as autograd
from torch.autograd import Variable

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class PSNR(nn.Module):
    """Layer to compute the PSNR loss between a pair of images
    """
    def __init__(self):
        super(PSNR, self).__init__()
        self.refl = nn.ReflectionPad2d(1)
 
    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mse = torch.mean((x - y) ** 2)
        return 10 * torch.log10(1 / mse)

class PSNRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(PSNRLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        diff = output - target
        mse = (diff * diff).mean(dim=(1, 2, 3))
        l1 = torch.abs(diff).mean(dim=(1, 2, 3))
        loss = torch.log(mse + self.eps + l1).mean()
        return loss

def Augment(Img, Label, Dark, CropSize=256, Prob=0.5):
    prob1 = random.random()
    prob2 = random.random()
    prob3 = random.random()
    prob4 = random.random()
    prob5 = random.random()
    prob6 = random.random()
    h, w, c = Img.shape
    h_start = random.randint(0, int(h-CropSize)) # random.randint(0, int(h*(1-CropSize)))
    w_start = random.randint(0, int(w-CropSize)) # random.randint(0, int(w*(1-CropSize)))
    Img = Img[h_start:h_start+CropSize, w_start:w_start+CropSize, :]
    Label = Label[h_start:h_start+CropSize, w_start:w_start+CropSize, :]
    Dark = Dark[h_start:h_start+CropSize, w_start:w_start+CropSize, :]
    # 左右翻转
    if prob1 < Prob:
        Img = cv2.flip(Img, 1)
        Label = cv2.flip(Label, 1)
        Dark = cv2.flip(Dark, 1)
    # 上下翻转
    if prob2 < Prob:
        Img = cv2.flip(Img, 0)
        Label = cv2.flip(Label, 0)
        Dark = cv2.flip(Dark, 0)
    # 旋转90度
    if prob3 < Prob:
        Img = np.rot90(Img)
        Label = np.rot90(Label)
        Dark = np.rot90(Dark)
    # # 对比度变换
    # if prob4 < Prob:
    #     Img = Img * random.uniform(0.8, 1.2)
    #     Dark = Dark * random.uniform(0.8, 1.2)
    #     Img = np.clip(Img, 0, 255)
    #     Dark = np.clip(Dark, 0, 255)
    # # 亮度变换
    # if prob5 < Prob:
    #     Img = Img + random.uniform(-10, 10)
    #     Dark = Dark + random.uniform(-10, 10)
    #     Img = np.clip(Img, 0, 255)
    #     Dark = np.clip(Dark, 0, 255)

    return Img.astype(np.uint8), Label.astype(np.uint8), Dark.astype(np.uint8)

def RandomMask(Img, mask_size=128):
    h, w, c = Img.shape
    h_start = random.randint(0, int(h-mask_size)) # random.randint(0, int(h*(1-CropSize)))
    w_start = random.randint(0, int(w-mask_size)) # random.randint(0, int(w*(1-CropSize)))
    Img[h_start:h_start+mask_size, w_start:w_start+mask_size, :] = 255
    return Img

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
	def __init__(self, requires_grad=False):
		super().__init__()
		vgg_pretrained_features = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()
		for x in range(2):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(2, 7):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(7, 12):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(12, 21):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(21, 30):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, X):
		h_relu1 = self.slice1(X)
		h_relu2 = self.slice2(h_relu1)
		h_relu3 = self.slice3(h_relu2)
		h_relu4 = self.slice4(h_relu3)
		h_relu5 = self.slice5(h_relu4)
		out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
		return out

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]


    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a[:, :3, ...]), self.vgg(p[:, :3, ...]), self.vgg(n[:, :3, ...])
        loss = 0
        for i in range(len(a_vgg)):
            lap = self.criterion(a_vgg[i], p_vgg[i])
            lan = self.criterion(a_vgg[i], n_vgg[i])
            loss += self.weights[i] * (lap - lan) # torch.log(lap / (lan + 1e-6))
        return loss

class WeightLoss(nn.Module):
    def __init__(self):
        super(WeightLoss, self).__init__()

    def forward(self, Weight, Dehazy, FreeHazy):
        return (torch.abs(Weight.mean(dim=1).unsqueeze(1)) * torch.abs(Dehazy - FreeHazy)).mean()


if __name__ == '__main__':
    PSNR = WeightLoss()
    x = torch.randn(1, 3, 512, 512)
    y = torch.randn(1, 3, 512, 512)
    print(PSNR(x, x, y))
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DehazyDataset(Dataset):
    def __init__(self, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        self.label_path = './Data/Label/'
        self.image_path = './Data/Raw/'
        self.img_list = os.listdir(self.image_path)
        self.img_list = self.img_list if mode == 'train' else [x for x in self.img_list if 'GT' not in x]
        random.shuffle(self.img_list)
        print('Loading data...')
        self.label_list = [cv2.imread(self.label_path + i) for i in self.img_list]
        self.img_list = [cv2.imread(self.image_path + i) for i in self.img_list]
        print('Data loaded.')

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, item):
        # img_name = self.img_list[item]
        # img = cv2.imread(self.image_path + img_name)
        # label = cv2.imread(self.label_path + img_name)
        img = self.img_list[item]
        label = self.label_list[item]
        if self.transform:
            data = self.transform(image=img, mask=label)
        return data['image'] / 255.0, data['mask'] / 255.0


if __name__ == '__main__':
    print('=========Testing Dataset========')
    train_transform = A.Compose([
        A.RandomCrop(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),

        ToTensorV2(transpose_mask=True)
    ])
    train = DataLoader(DehazyDataset(transform=train_transform, mode='train'))
    for img, label in train:
        print(img.shape, label.shape)
        # print(img.max(), img.min())
        # break
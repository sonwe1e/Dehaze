import os
import cv2
import numpy as np

LabelPath = './Data/Label/'

# 将LabelPath中的图片转换为灰度图并保存在./Data/Grey/中
def Grey():
    for filename in os.listdir(LabelPath):
        img = cv2.imread(LabelPath + filename, 0)
        cv2.imwrite('./Data/Grey/' + filename, img)

if __name__ == '__main__':
    Grey()
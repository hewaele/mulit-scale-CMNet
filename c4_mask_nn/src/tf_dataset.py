"""
function: 制作tfdata数据集解决oom问题
time:2019年12月06日10:40:02
author： hewa

"""

import tensorflow as tf
import numpy as np
import os
import time
import sys
import re

#过滤图片选择用于训练的图片及mask
def filter_image(image_path):
    m_rep = re.compile('\d{3}_B')
    t_rep = re.compile('\d{3}_F')
    image_list = sorted(os.listdir(image_path))
    image_list = [v for v in image_list if m_rep.match(v) or t_rep.match(v)]
    mask_image = []
    train_image = []

    for image in image_list:
        if m_rep.match(image):
            mask_image.append(image)
        else:
            train_image.append(image)
    mask_image = mask_image*25
    mask_image = sorted(mask_image)

    # print(len(mask_image))
    # print(len(train_image))
    # for i, j in zip(mask_image, train_image):
    #     print('{}  {}'.format(i, j))
    return train_image, mask_image
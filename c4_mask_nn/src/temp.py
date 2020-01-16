from tf_dataset import filter_image
from utiles import my_generator
import numpy as np
from matplotlib import pyplot as plt
from casia_data_process import get_casiadataset


target_path = '/home/hewaele/PycharmProjects/creat_cmfd_image/cmfd_data/images_small'
mask_path = '/home/hewaele/PycharmProjects/creat_cmfd_image/cmfd_data/mask_small'
x_list, y_list = get_casiadataset(target_path, mask_path)
print('start')
for x, y in zip(x_list, y_list):
    x = x.split('_')[-1]
    y = y.split('_')[-1]
    print(x, y)
    if x != y:
        print('error: x:{} y:{}'.format(x, y))
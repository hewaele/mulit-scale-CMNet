import os

image_path = '../data/CoMoFoD_small'
from tf_dataset import filter_image
extra_dic = {}

x, y = filter_image(image_path)
for i in y:
    e = i.split('.')[-1]
    if e in extra_dic:
        extra_dic[e] += 1
    else:
        extra_dic[e] = 1

print(extra_dic)
from tf_dataset import filter_image
from utiles import my_generator
import numpy as np

image_path = '../data/CoMoFoD_small/'
x_list, y_list = filter_image(image_path)
print(x_list[::25])
print(y_list[::25])
test = my_generator(x_list[::25], y_list[::25], 1, new_size=256)
count = 0
flag = 0
for ti in test:
    print(ti)
    count += 1
    if count >= 0:
        break
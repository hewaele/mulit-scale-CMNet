from tf_dataset import filter_image
from utiles import my_generator
import numpy as np

image_path = '../data/CoMoFoD_small/'
x_list, y_list = filter_image(image_path)
print(x_list[::25])
print(y_list[::25])
test = my_generator(x_list[::25], y_list[::25], 2, new_size=256)
count = 0
flag = 0
for ti in test:
    print(count)
    flag += np.sum(ti[1])
    count += 1
    if count >= 200:
        break

print(flag/(256*256*count))
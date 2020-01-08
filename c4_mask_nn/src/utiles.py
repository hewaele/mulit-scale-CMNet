import numpy as np
from PIL import Image

def my_generator(x_list, y_list, batchs, new_size=256, rescale = True):
    x = np.zeros([batchs, new_size, new_size, 3])
    y = np.zeros([batchs, new_size, new_size, 1])
    b = 0
    while 1:
        for image_path, mask_path in zip(x_list, y_list):
            img = Image.open(image_path).convert('RGB').resize([new_size, new_size])
            mask = Image.open(mask_path).convert('L').resize([new_size, new_size])
            img = np.array(img).reshape([new_size, new_size, 3])
            mask = np.array(mask).reshape([new_size, new_size, 1])
            if rescale:
                img = np.multiply(img, 1/255.0)
            mask = np.multiply(mask, 1/255.0)
            # 判断是否进行标签转换
            mask = np.round(mask)
            x[b] = img
            y[b] = mask
            b += 1
            if b >= batchs:
                b = 0
                yield x, y
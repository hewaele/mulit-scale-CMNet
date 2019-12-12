import tensorflow as tf
import numpy as np
import os
import re
from PIL import Image
from matplotlib import pyplot as plt


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

def image_preprocess(image_path, train_x, train_y):
    #将mask转换为二值
    train = []
    mask = []
    for img_name in train_x[:1000]:
        img = Image.open(os.path.join(image_path, img_name)).convert('RGB').resize([256, 256])
        train.append(np.array(img)/255)
        assert np.array(img).shape == (256, 256, 3)

    for img in train_y[:1000]:
        img = Image.open(os.path.join(image_path, img)).convert('L').resize([256, 256])
        mask.append(np.array(img)/255)

    x = np.array(train).reshape([-1, 256, 256, 3])
    y = np.array(mask).reshape([-1, 256, 256, 1])
    return x, y


if __name__ == "__main__":
    image_path = '../data/CoMoFoD_small'
    train_image, mask_image  = filter_image(image_path)

    # img = Image.open(os.path.join(image_path, mask_image[0])).convert('L')
    # img2 = Image.open(os.path.join(image_path, train_image[0]))
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(img2)
    # plt.show()
    # c = img.resize([256, 256])
    # Image.Image.show(c)

    x, y = image_preprocess(image_path, train_image, mask_image)
    # print(np.array(train).reshape(-1, 256, 256, 3)/255)
    # print(np.array(mask).reshape(-1, 256, 256, 1)/255)










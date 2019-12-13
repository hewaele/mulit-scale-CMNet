import os
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image


def creat_mask_image_list(mask_path, txt_path):
    #创建一个文件
    with open(txt_path, 'w') as fb:
        for img in os.listdir(mask_path):
            fb.writelines(img+'\n')

def get_target_image_list(mask_path, source_path):
    """
    寻找source原始图片中存在的图片
    :param source_path:
    :param image_list_path:
    :return: image_path_list
    """
    mask_image_list = []
    for img in os.listdir(mask_path):
        mask_image_list.append(img.split('.')[0])
    target_image_list = []
    for img in os.listdir(source_path):
        if img.split('.')[0] in mask_image_list:
            target_image_list.append(img)

    return target_image_list

#通过读取image_list 将用于训练的图片添加到source文件夹中
def creat_target_images(source_path, mask_path, target_path):
    #读取mask_path 中的文件
    target_images_list = get_target_image_list(mask_path, source_path)
    #将图片另存到另一个文件夹
    for img_name in target_images_list[:]:
        img = Image.open(os.path.join(source_path, img_name))
        img.save(os.path.join(target_path, img_name.split('.')[0] + '.png'))

def show_test(target_path, mask_path):
    images_list = sorted(os.listdir(target_path))
    for img in images_list[:3]:
        plt.figure()
        plt.subplot(121)
        img1 = Image.open(os.path.join(target_path, img)).convert('RGB')
        plt.imshow(img1)
        plt.subplot(122)
        img2 = Image.open(os.path.join(mask_path, img)).convert('L')
        print(np.array(img2), '\n')
        plt.imshow(img2)
        plt.show()

#将mask文件转话为二值并保存
def creat_mask_images(source_mask_path, target_mask_path, target_mask_list):
    for img_name in target_mask_list[:]:
        img_name = img_name.split('.')[0]+'.png'
        img = Image.open(os.path.join(source_mask_path, img_name))
        #转换为二值
        img = img.convert('L')
        img = np.array(img)
        img = img - 29
        img = np.logical_not(np.logical_not(img)).astype(np.uint8)
        # plt.imshow(img)
        # plt.show()
        # print(np.sum(img))
        img = Image.fromarray(img*255)
        img.save(os.path.join(target_mask_path, img_name))

def get_casiadataset(target_path, mask_path):
    target_list = [os.path.join(target_path, i) for i in os.listdir(target_path)]
    mask_list = [os.path.join(mask_path, i) for i in os.listdir(mask_path)]
    target_list = sorted(target_list)
    mask_list = sorted(mask_list)

    return target_list, mask_list

def main():
    #根据mask 获取图片
    mask_path = '../data/casia-dataset/GT_Mask'
    source_path = '../data/casia-dataset/Tp'
    target_path = '../data/casia-dataset/target'
    target_mask_path = '../data/casia-dataset/mask'
    # creat_target_images(source_path, mask_path, target_path)
    # show_test(target_path, mask_path)
    target_mask_list = get_target_image_list(mask_path, source_path)
    creat_mask_images(mask_path, target_mask_path, target_mask_list)

if __name__ == '__main__':
    main()
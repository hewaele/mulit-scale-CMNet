from casia_data_process import get_casiadataset
target_path = '../data/augmentation_data/image'
mask_path = '../data/augmentation_data/mask'
x_list, y_list = get_casiadataset(target_path, mask_path)

for x, y in zip(x_list, y_list):
    x = x.split('/')[-1]
    y = y.split('/')[-1]
    if x == y:
        print(x, y)
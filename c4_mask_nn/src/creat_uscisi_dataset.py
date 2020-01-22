#%%
from uscisi_api import USCISI_CMD_API
from matplotlib import pyplot
import os
import numpy as np
from PIL import Image
import sys
#%% md
# Sample USC-ISI CMD Dataset
#%%
image_path = '../data/uscisi_dataset/images'
mask_path = '../data/uscisi_dataset/mask'

lmdb_dir = 'C:/aha/USCISI-CMFD'
dataset = USCISI_CMD_API( lmdb_dir=lmdb_dir,
                          sample_file=os.path.join(lmdb_dir, 'samples.keys' ),
                          differentiate_target=True )
#%% md
# Retrieve the first 24 samples in the dataset
#%%
samples = dataset(range(100000))

count = 0
for si in samples:

    count += 1
    try:
        print(count)
        image = si[0]
        mask = si[1]
        image = Image.fromarray(image)
        mask = np.round(np.logical_not(mask[:, :, 2])) * 255
        mask = Image.fromarray(np.int8(mask)).convert('L')
        image.save(os.path.join(image_path, 'image_' + str(count) + '.png'))
        mask.save(os.path.join(mask_path, 'mask_' + str(count) + '.png'))
    except:
        pass

    if count == 100000:
        break


# mask = Image.fromarray(np.uint8(mask)).convert('L')
# print(np.array(mask))
#
# image.show()
# mask.show()
# dataset.visualize_samples( samples )
# #%% md
# # Retrieve 24 random samples in the dataset
# #%%
# print('hrer')
# samples = dataset( [None]*24 )
#
# dataset.visualize_samples( samples )
# #%% md
# # Get the exact 50th sample in the dataset
# #%%
# sample = dataset[50]
# dataset.visualize_samples( [sample] )
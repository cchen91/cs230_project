from __future__ import print_function

import os
import numpy as np

import sys
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

data_path = '/data2/yeom/ky_fetal/cs230_data/train/ax/'
image_path = data_path+'images/'

nh = 512
nw = 512
nc = 1
ratio_tr = 0.8

def create_train_data():
    
    x1 = np.load(data_path+"x_ax_2d.npy")
    y1 = np.load(data_path+"y_ax_2d.npy")
    x2 = np.load(data_path+"x_cor_2d.npy")
    y2 = np.load(data_path+"y_cor_2d.npy")
    x3 = np.load(data_path+"x_sag_2d.npy")
    y3 = np.load(data_path+"y_sag_2d.npy")

    nm1 = x1.shape[0]
    nm2 = x2.shape[0]
    nm3 = x3.shape[0]
    nm1tr = int(nm1*ratio_tr)
    nm2tr = int(nm2*ratio_tr)
    nm3tr = int(nm3*ratio_tr)
    nm1te = nm1-nm1tr
    nm2te = nm2-nm2tr
    nm3te = nm3-nm3tr
    nmtr = nm1tr+nm2tr+nm3tr
    nmte = nm1te+nm2te+nm3te

    x1 = x1/x1.max(axis = (1, 2)).reshape((nm1, 1, 1))
    x1.astype('float32')
    x2 = x2/x2.max(axis = (1, 2)).reshape((nm2, 1, 1))
    x2.astype('float32')
    x3 = x3/x3.max(axis = (1, 2)).reshape((nm3, 1, 1))
    x3.astype('float32')
    
    x_tr = np.ndarray((nmtr, nh, nw, 1), dtype=np.float32)
    y_tr = np.ndarray((nmtr, nh, nw, 1), dtype=np.float32)
    z_tr = np.zeros((nmtr, nh, nw, 3))#visualization
    x_te = np.ndarray((nmte, nh, nw, 1), dtype=np.float32)
    y_te = np.ndarray((nmte, nh, nw, 1), dtype=np.float32)
    z_te = np.zeros((nmte, nh, nw, 3))#visualization
    
    print('-'*30)
    print('Creating data set')
    print('-'*30)

    x_tr[0:nm1tr, ..., 0] = x1[0:nm1tr, ...]
    x_tr[nm1tr:nm1tr+nm2tr, ..., 0] = x2[0:nm2tr, ...]
    x_tr[nm1tr+nm2tr:nmtr, ..., 0] = x3[0:nm3tr, ...]
    x_te[0:nm1te, ..., 0] = x1[nm1tr:nm1, ...]
    x_te[nm1te:nm1te+nm2te, ..., 0] = x2[nm2tr:nm2, ...]
    x_te[nm1te+nm2te:nmte, ..., 0] = x3[nm3tr:nm3, ...]
    y_tr[0:nm1tr, ..., 0] = y1[0:nm1tr, ...]
    y_tr[nm1tr:nm1tr+nm2tr, ..., 0] = y2[0:nm2tr, ...]
    y_tr[nm1tr+nm2tr:nmtr, ..., 0] = y3[0:nm3tr, ...]
    y_te[0:nm1te, ..., 0] = y1[nm1tr:nm1, ...]
    y_te[nm1te:nm1te+nm2te, ..., 0] = y2[nm2tr:nm2, ...]
    y_te[nm1te+nm2te:nmte, ..., 0] = y3[nm3tr:nm3, ...]

    z_tr[..., 0] = x_tr[..., 0]
    z_tr[..., 2] = y_tr[..., 0]
    z_te[..., 0] = x_te[..., 0]
    z_te[..., 2] = y_te[..., 0]
    z_tr.astype('float32')
    z_te.astype('float32')
    
    np.save(data_path+'x_tr.npy', x_tr)
    np.save(data_path+'y_tr.npy', y_tr)
    np.save(data_path+'z_tr.npy', z_tr)
    np.save(data_path+'x_te.npy', x_te)
    np.save(data_path+'y_te.npy', y_te)
    np.save(data_path+'z_te.npy', z_te)
    
    """
    for i in range(nmtr):
        mpl.image.imsave(image_path+"train/x_"+str(i+1)+".png", x_tr[i, ..., 0], cmap = 'gray')
        mpl.image.imsave(image_path+"train/y_"+str(i+1)+".png", y_tr[i, ..., 0], cmap = 'gray')
        mpl.image.imsave(image_path+"train/z_"+str(i+1)+".png", z_tr[i, ...])
    for i in range(nmte):
        mpl.image.imsave(image_path+"test/x_"+str(i+1)+".png", x_te[i, ..., 0], cmap = 'gray')
        mpl.image.imsave(image_path+"test/y_"+str(i+1)+".png", y_te[i, ..., 0], cmap = 'gray')
        mpl.image.imsave(image_path+"test/z_"+str(i+1)+".png", z_te[i, ...])
    """
    print('Saving to .npy files done.')

def load_fetal_data():
    x_tr = np.load(data_path+'x_tr.npy')
    y_tr = np.load(data_path+'y_tr.npy')
    x_te = np.load(data_path+'x_te.npy')
    y_te = np.load(data_path+'y_te.npy')
    return x_tr, y_tr, x_te, y_te

"""
def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)
    
    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    imgs_id = np.ndarray((total, 1), dtype=np.int32)
    
    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        
        imgs[i, ..., 0] = img
        imgs_id[i, 0] = img_id
        
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    print('Loading done.')

    np.save(save_path+'imgs_test.npy', imgs)
    np.save(save_path+'imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')

def load_test_data():
    imgs_test = np.load(save_path+'imgs_test.npy')
    imgs_id = np.load(save_path+'imgs_id_test.npy')
    return imgs_test, imgs_id
"""

if __name__ == '__main__':
    create_train_data()
# create_test_data()


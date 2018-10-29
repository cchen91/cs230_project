from model import *
from data import *

x_tr, y_tr, x_te, y_te = load_fetal_data()
model = unet()
model = load_model('model_100ep.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

yhat_tr = model.predict(x_tr, verbose=1)
yhat_tr = np.floor(yhat_tr+0.5)
np.save(data_path+'yhat_tr.npy', yhat_tr)
yhat_te = model.predict(x_te, verbose=1)
yhat_te = np.floor(yhat_te+0.5)
np.save(data_path+'yhat_te.npy', yhat_te)

nmtr, nh, nw, nc = x_tr.shape
zhat_tr = np.ndarray((nmtr, nh, nw, 3))
zhat_tr[..., 0] = x_tr[..., 0]
zhat_tr[..., 2] = yhat_tr[..., 0]
zhat_tr.astype('float32')

nmte, nh, nw, nc = x_te.shape
zhat_te = np.ndarray((nmte, nh, nw, 3))
zhat_te[..., 0] = x_te[..., 0]
zhat_te[..., 2] = yhat_te[..., 0]
zhat_te.astype('float32')

"""
for i in range(nmtr):
    mpl.image.imsave(image_path+"/train/yhat_"+str(i+1)+".png", yhat_tr[i, ..., 0], cmap = 'gray')
    mpl.image.imsave(image_path+"/train/zhat_"+str(i+1)+".png", zhat_tr[i, ...])

for i in range(nmte):
    mpl.image.imsave(image_path+"/test/yhat_"+str(i+1)+".png", yhat_te[i, ..., 0], cmap = 'gray')
    mpl.image.imsave(image_path+"/test/zhat_"+str(i+1)+".png", zhat_te[i, ...])
"""

y_tr.astype(bool)
yhat_tr.astype(bool)
y_te.astype(bool)
yhat_te.astype(bool)

#y_tr_sum = np.sum(y_tr, axis = (1, 2))
#yhat_tr_sum = np.sum(yhat_tr, axis = (1, 2))

inter_tr = np.sum(np.logical_and(y_tr, yhat_tr), axis = (1, 2))+0.001
union_tr = np.sum(np.logical_or(y_tr, yhat_tr), axis = (1, 2))+0.001
iou_tr = inter_tr/union_tr
#print(inter_tr)
#print(union_tr)
#print('IoU score of training set:\n\r')
#print(iou_tr)
print('\n\r IoU score mean of training set:\n\r')
print(iou_tr.mean())

inter_te = np.sum(np.logical_and(y_te, yhat_te), axis = (1, 2))+0.001
union_te = np.sum(np.logical_or(y_te, yhat_te), axis = (1, 2))+0.001
iou_te = inter_te/union_te
#print(inter_te)
#print(union_te)
#print('IoU score of testing set:\n\r')
#print(iou_te)
print('\n\r IoU score mean of testing set:\n\r')
print(iou_te.mean())


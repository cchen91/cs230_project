from model import *
from data import *

x_tr, y_tr, x_te, y_te = load_fetal_data()
model = unet()
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)

model.fit(x_tr, y_tr, batch_size=4, epochs=20, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_te, y_te))
model.save('model_20ep.h5')

"""
model.fit(x_tr, y_tr, batch_size=8, epochs=10, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_te, y_te))
model.save('model_10ep.h5')
"""
for i in range(4):
    model = load_model('model_'+str(20*(i+1))+'ep.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    model.fit(x_tr, y_tr, batch_size=4, epochs=20, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_te, y_te))
    model.save('model_'+str(20*(i+2))+'ep.h5')


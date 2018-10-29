import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

smooth = 1

def unet(pretrained_weights = None,input_size = (512, 512, 1)):
    X = Input(input_size)
    
    #Stage1
    conv1 = Conv2D(32, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(X)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    #Stage2
    conv2 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    #Stage3
    conv3 = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    #Stage4
    conv4 = Conv2D(256, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    #Stage5
    conv5 = Conv2D(512, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    #Stage6
    up6 = Conv2D(256, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis=3)
    conv6 = Conv2D(256, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    #Stage7
    up7 = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    #Stage8
    up8 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    #Stage9
    up9 = Conv2D(32, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(32, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    #Stage10
    conv10 = Conv2D(1, kernel_size=(1, 1), activation = 'sigmoid')(conv9)

    model = Model(inputs = X, outputs = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = [dice_coef])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


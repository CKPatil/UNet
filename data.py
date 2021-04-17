#!/usr/bin/env python
import os
import tensorflow as tf 
import numpy as np 
from tqdm import tqdm
from skimage.io import imread,imshow
from skimage.transform import resize
import matplotlib.pyplot as plt 
import random
import cv2

seed=42
np.random.seed=seed 

#Initialize the size of the testing and training images
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

#Define the path where the images are stored
TRAIN_PATH='C:/Users/ChetanKPatil/Desktop/unet/data/train/image/'
LABEL_PATH='C:/Users/ChetanKPatil/Desktop/unet/data/train/label/'
TEST_PATH='C:/Users/ChetanKPatil/Desktop/unet/data/test/'

#Checking where the images exists within TRAIN_PATH,LABEL_PATH and TEST_PATH
trainid=next(os.walk(TRAIN_PATH))[2]
labelid=next(os.walk(TRAIN_PATH))[2]
testid=next(os.walk(TEST_PATH))[2]

#Initialize training and testing data to 0
xtrain=np.zeros((len(trainid),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
ytrain=np.zeros((len(trainid),IMG_HEIGHT,IMG_WIDTH,1),dtype=np.bool)
xtest=np.zeros((len(trainid),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
# print(xtest)

#Copy training image data from TRAIN_PATH
# print("xtrain = ")
for n,filename in enumerate(trainid):
	path=TRAIN_PATH+filename
	xtrain[n]=cv2.imread(path)[:,:,:IMG_CHANNELS]
	# print(xtrain)

#Copy training mask data from LABEL_PATH
# print("ytrain = ")
for n,filename in enumerate(labelid):
	path=LABEL_PATH+filename
	ytrain[n]=cv2.imread(path)[:,:,:1]
	# print(ytrain)

#Copy test image data from TEST_PATH
# print("xtest = ")
for n,filename in enumerate(testid):
	path=TEST_PATH+filename
	xtest[n]=cv2.imread(path)[:,:,:IMG_CHANNELS]
	# print(xtest)

# imshow(xtest[29])
# plt.show()
# imagex=random.randint(0,len(trainid))
# imshow(xtrain[imagex])
# plt.show()
# imshow(ytrain[imagex])
# plt.show()
# imshow(xtest[imagex])
# plt.show()

#####################################################################################################################
#Building UNet model

inputs = tf.keras.layers.Input((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))
s=tf.keras.layers.Lambda(lambda x: x/255)(inputs)

#Contraction Path
conv1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(s)
conv1=tf.keras.layers.Dropout(0.1)(conv1)
conv1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv1)
pool1=tf.keras.layers.MaxPooling2D((2,2))(conv1)

conv2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool1)
conv2=tf.keras.layers.Dropout(0.1)(conv2)
conv2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv2)
pool2=tf.keras.layers.MaxPooling2D((2,2))(conv2)

conv3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool2)
conv3=tf.keras.layers.Dropout(0.2)(conv3)
conv3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv3)
pool3=tf.keras.layers.MaxPooling2D((2,2))(conv3)

conv4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool3)
conv4=tf.keras.layers.Dropout(0.2)(conv4)
conv4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv4)
pool4=tf.keras.layers.MaxPooling2D((2,2))(conv4)

conv5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool4)
conv5=tf.keras.layers.Dropout(0.3)(conv5)
conv5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv5)

#Expansive Path
u6=tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2), padding='same')(conv5)
u6=tf.keras.layers.concatenate([u6,conv4])
conv6=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u6)
conv6=tf.keras.layers.Dropout(0.2)(conv6)
conv6=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv6)

u7=tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2), padding='same')(conv6)
u7=tf.keras.layers.concatenate([u7,conv3])
conv7=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u7)
conv7=tf.keras.layers.Dropout(0.2)(conv7)
conv7=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv7)

u8=tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2), padding='same')(conv7)
u8=tf.keras.layers.concatenate([u8,conv2])
conv8=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u8)
conv8=tf.keras.layers.Dropout(0.1)(conv8)
conv8=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv8)

u9=tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2), padding='same')(conv8)
u9=tf.keras.layers.concatenate([u9,conv1])
conv9=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u9)
conv9=tf.keras.layers.Dropout(0.1)(conv9)
conv9=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv9)

outputs=tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(conv9)

model = tf.keras.Model(inputs = [inputs], outputs = [outputs])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet.hdf5',verbose=1, save_best_only=True)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=75, monitor='val_loss'),tf.keras.callbacks.TensorBoard(log_dir='logs')]
model.fit(xtrain, ytrain, validation_split=0.1,batch_size=1, epochs=100, callbacks=callbacks)

# model.summary()
##################################################################################################################
#Running Predictions

idx = random.randint(0, len(xtrain))


preds_train = model.predict(xtrain[:int(xtrain.shape[0]*0.9)], verbose=1)
preds_val = model.predict(xtrain[int(xtrain.shape[0]*0.9):], verbose=1)
preds_test = model.predict(xtest, verbose=1)

# print(preds_train)
 
preds_train_t = (preds_train > 0.7).astype(np.bool)
preds_val_t = (preds_val > 0.7).astype(np.bool)
preds_test_t = (preds_test > 0.7).astype(np.bool)


# Perform a check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(xtrain[ix])
plt.show()
imshow(np.squeeze(ytrain[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(xtrain[int(xtrain.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(ytrain[int(ytrain.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

# Perform a check on some random test samples
ix = random.randint(0, len(preds_test_t))
imshow(xtrain[int(xtrain.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(ytrain[int(ytrain.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()

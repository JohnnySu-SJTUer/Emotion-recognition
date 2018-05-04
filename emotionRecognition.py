# # ******************************* Keras FER2013 Emotion Recognition ************************************
from keras.datasets import mnist
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Flatten
from keras.optimizers import Adam
import pickle


with open('./datasets/train.pkl', 'rb') as f:
    train_imgs, train_labels = pickle.load(f)
with open('./datasets/valid.pkl', 'rb') as f:
    valid_imgs, valid_labels = pickle.load(f)
with open('./datasets/test.pkl', 'rb') as f:
    test_imgs, test_labels = pickle.load(f)
# data pre-processing
train_labels = np_utils.to_categorical(train_labels, num_classes=7)
valid_labels = np_utils.to_categorical(valid_labels, num_classes=7)
test_labels = np_utils.to_categorical(test_labels, num_classes=7)
trainval_imgs = np.concatenate([train_imgs, valid_imgs], axis=0)
trainval_labels = np.concatenate([train_labels, valid_labels], axis=0)

model = Sequential()
# conv1 layer
model.add(Convolution2D(filters = 128, kernel_size = (4, 4), border_mode = 'same', input_shape = (1, 48, 48))) #1：channels; 48:尺寸
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.8))
# conv2 layer
model.add(Convolution2D(filters = 128, kernel_size = (4, 4), border_mode = 'same', input_shape = (128, 48, 48))) #1：channels; 48:尺寸
model.add(BatchNormalization())
model.add(Activation('relu'))
# conv3 layer
model.add(Convolution2D(filters = 128, kernel_size = (4, 4), border_mode = 'same', input_shape = (128, 48, 48))) #1：channels; 48:尺寸
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), border_mode = 'same'))
model.add(Dropout(0.8))
# conv4 layer
model.add(Convolution2D(filters = 128, kernel_size = (4, 4), border_mode = 'same', input_shape = (128, 24, 24))) #1：channels; 24:尺寸
model.add(BatchNormalization())
model.add(Activation('relu'))
# conv5 layer
model.add(Convolution2D(filters = 128, kernel_size = (4, 4), border_mode = 'same', input_shape = (128, 24, 24))) #1：channels; 24:尺寸
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), border_mode = 'same'))
model.add(Dropout(0.8))
# conv6 layer
model.add(Convolution2D(filters = 128, kernel_size = (4, 4), border_mode = 'same', input_shape = (128, 12, 12))) #1：channels; 12:尺寸
model.add(BatchNormalization())
model.add(Activation('relu'))
# conv7 layer
model.add(Convolution2D(filters = 128, kernel_size = (4, 4), border_mode = 'same', input_shape = (128, 12, 12))) #1：channels; 12:尺寸
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), border_mode = 'same'))
model.add(Dropout(0.8))
model.add(Flatten())
# full connected layer 1
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.8))
# full connected layer 2
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.8))
# output layer
model.add(Dense(7))
model.add(Activation('softmax'))

adam = Adam(lr = 1e-3)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(trainval_imgs, trainval_labels, epochs=100, batch_size=512, validation_split=0.1, shuffle=True)

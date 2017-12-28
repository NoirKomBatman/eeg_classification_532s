import sys
import numpy as np
import scipy as sp
import math
from scipy.io import loadmat
from scipy.signal import medfilt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling1D, Conv1D, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

max_len = 96735
channel_num = 25
epochs = 3

def create_CNN():
    global max_len
    global channel_num
    print('Build CNN model...')

    if K.image_data_format() == 'channels_first':
        shape_ord = (1, 60000)
    else:
        shape_ord = (60000, 1)

    model = Sequential((
        Conv1D(32, 10, activation='relu', input_shape=shape_ord),
        Conv1D(64,  10, activation='relu'),
        MaxPooling1D(pool_size=10),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='sigmoid'),
    ))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_perceptron():
    global max_len
    global channel_num
    print('Build Perceptron model...')

    if K.image_data_format() == 'channels_first':
        shape_ord = (1, 60000)
    else:
        shape_ord = (60000, 1)

    model = Sequential()

    model.add(Flatten(input_shape=shape_ord))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def load_data(v):
    global max_len
    subjects = [1,2,3,4,5,6,7,8,9]
    X = []
    Y = []
    count = 0
    data_dir = '../data/four_class_motor_imagery'
    for i in subjects:
        D = loadmat(data_dir + '/P%02d' % i + v + '.mat' )
        sessions = D['sessions']
        for sess in sessions:
            X_sess = sess[0][0]
            if X_sess.shape[0] != max_len:
                continue
                ix = X_sess.shape[0]
                X_sess = np.vstack([X_sess, np.zeros((max_len - X_sess.shape[0], channel_num))])
            if X_sess.shape[0] > max_len:
                continue
            gender = sess[0][1][0]
            if gender == 'male':
                gender = 0
            else:
                gender = 1
            print(X_sess.shape)
            print(gender)
            X.append(X_sess.T)
            Y.append([gender for x in range(25)])
            count+=1
    return np.array(X).reshape(count*channel_num,96735), np.array(Y).reshape(count*channel_num,1)


def info(type, value, tb):
   if hasattr(sys, 'ps1') or not sys.stderr.isatty():
      sys.__excepthook__(type, value, tb)
   else:
      import traceback, pdb
      traceback.print_exception(type, value, tb)
      print
      pdb.pm()

def normalize(X):
    return (X - np.min(X))/(np.max(X) + np.min(X))

def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


model = create_CNN()
#model = create_perceptron()

sys.excepthook = info

seed = 7
np.random.seed(seed) # for reproducibility

# get training data
X_train, Y_train = load_data('T')
n_train = X_train.shape[0]

# split into train/validation
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.33, random_state=seed)
Y_train = to_categorical(Y_train, 2).reshape((Y_train.shape[0],2))
Y_valid = to_categorical(Y_valid, 2).reshape((Y_valid.shape[0],2))

# remove outliers
X_train[X_train>80] = 80
X_train[X_train<-80] = -80

# mean and absolute min/max normalization across channels
for i in range(15):
    X_train[i,:] = X_train[i,:] - np.mean(X_train[i,:])
    ma = np.max(X_train[i,:])
    mi = np.min(X_train[i,:])
    if np.abs(ma) > np.abs(mi):
        X_train[i,:] = X_train[i,:]/np.abs(ma)
    else:
        X_train[i,:] = X_train[i,:]/np.abs(mi)

# median filtering
X_train = medfilt(X_train)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
#X_train = moving_average(X_train)

# get testing data
X_test, Y_test = load_data('E')
n_test = X_test.shape[0]
Y_test = to_categorical(Y_test, 2).reshape((Y_test.shape[0],2))

# outlier removal
X_test[X_test>80] = 80
X_test[X_test<-80] = -80

# mean and max/min normalization across channels
for i in range(15):
    X_test[i,:] = X_test[i,:] - np.mean(X_test[i,:])
    ma = np.max(X_test[i,:])
    mi = np.min(X_test[i,:])
    if np.abs(ma) > np.abs(mi):
        X_test[i,:] = X_test[i,:]/np.abs(ma)
    else:
        X_test[i,:] = X_test[i,:]/np.abs(mi)

# median filtering
X_test = medfilt(X_test)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
#X_test = moving_average(X_test)

# step decay for gradient descent
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

#X_train = X_train[:,20000:60000]
#X_valid = X_valid[:,20000:60000]
#X_test = X_test[:,20000:60000]
print(X_train.shape,Y_train.shape)
print(X_valid.shape,Y_valid.shape)
print(X_test.shape, Y_test.shape)

# train
hist = model.fit(np.expand_dims(X_train,axis=2), Y_train, epochs=epochs, validation_data=(np.expand_dims(X_valid,axis=2), Y_valid))

# print training results
fig_loss = plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Training', 'Validation'])

fig_acc = plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['Training', 'Validation'], loc='lower right')

# evaluating the model on the test data
loss, accuracy = model.evaluate(np.expand_dims(X_test,axis=2), Y_test, verbose=0)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

fig_loss.savefig('fig_loss.png')
fig_acc.savefig('fig_acc.png')

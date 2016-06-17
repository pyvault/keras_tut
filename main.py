from keras.layers import Input, Dense
from keras.models import Model
from dataacq import getset
from custom_noise import salt_and_pepper_custom
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = Input(shape=(28900,))
encoded = Dense(4800, activation='relu')(input_img)
encoded = Dense(2400, activation='relu')(encoded)
encoded = Dense(1200, activation='relu')(encoded)

decoded = Dense(2400, activation='relu')(encoded)
decoded = Dense(4800, activation='relu')(decoded)
decoded = Dense(28900, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np
import random

x_train = getset('data/Train/Frontalized')
x_test = getset('data/Test/Frontalized')
x_train_noisy = getset('data/Train/Posed')
x_test_noisy = getset('data/Test/Posed')
"""
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train_noisy = salt_and_pepper_custom(x_train)
x_test_noisy = salt_and_pepper_custom(x_test)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

x_train_noisy = np.reshape(x_train_noisy, (len(x_train_noisy), 784))
x_test_noisy = np.reshape(x_test_noisy, (len(x_test_noisy), 784))
"""

autoencoder.fit(x_train_noisy, x_train,
                nb_epoch=150,
                batch_size=10,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# encode and decode some digits
# note that we take them from the *test* set

decoded_imgs = autoencoder.predict(x_test_noisy)
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test_noisy[i].reshape(170, 170))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(170, 170))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

"""
#noise generation alternative
#train
saltarr = np.ones((28, 28)) #ones
saltarr2 = np.zeros((28, 28)) #zeros
#test
saltarr_te = np.ones((28, 28)) #ones
saltarr2_te = np.zeros((28, 28)) #zeros
dim = 28
pixs = 5
range_lim = 1 #white
range_lim2 = 0 #black

rdarray_x = random.sample(range(0, dim - pixs), range_lim)
rdarray_x2 = random.sample(range(10, 20), range_lim)

rdarray_x_te = random.sample(range(0, dim - pixs), range_lim)
rdarray_x2_te = random.sample(range(10, 20), range_lim)

for x in range(0, range_lim):
    saltarr[rdarray_x[x]:rdarray_x[x] + pixs, rdarray_x2[x]:rdarray_x2[x] + pixs] = 0  # 0
for y in range(0, range_lim):
    saltarr2[rdarray_x[y]:rdarray_x[y] + pixs, rdarray_x2[y]:rdarray_x2[y] + pixs] = 1  # 1

for x in range(0, range_lim):
    saltarr_te[rdarray_x_te[x]:rdarray_x_te[x] + pixs, rdarray_x2_te[x]:rdarray_x2_te[x] + pixs] = 0  # 0
for y in range(0, range_lim):
    saltarr2_te[rdarray_x_te[y]:rdarray_x_te[y] + pixs, rdarray_x2_te[y]:rdarray_x2_te[y] + pixs] = 1  # 1

saltzero = np.logical_not(saltarr) * saltarr2
saltzero_te = np.logical_not(saltarr_te) * saltarr2_te
#noise_factor = 0.5
x_train_noisy = x_train2 * saltarr + saltzero
x_test_noisy = x_test2 * saltarr + saltzero
"""
import numpy as np
import random
from keras.datasets import mnist
import matplotlib.pyplot as plt

def salt_and_pepper_custom(input):

    dim = 28
    pixs = 5
    batch = 100
    range_lim = 1  # white
    range_lim2 = 0  # black
    w = 60000
    inner_c = 0
    for i in range(batch, w+batch, batch):

        saltarr = np.ones((28, 28))  # ones
        saltarr2 = np.zeros((28, 28))  # zeros

        rdarray_x = random.sample(range(0, dim - pixs), range_lim)
        rdarray_x2 = random.sample(range(10, 20), range_lim)

        # print rdarray_x
        # print rdarray_x2

        for x in range(0, range_lim):
            saltarr[rdarray_x[x]:rdarray_x[x] + pixs, rdarray_x2[x]:rdarray_x2[x] + pixs] = 0  # 0
        for y in range(0, range_lim):
            saltarr2[rdarray_x[y]:rdarray_x[y] + pixs, rdarray_x2[y]:rdarray_x2[y] + pixs] = 1  #

        saltzero = np.logical_not(saltarr) * saltarr2
        input[inner_c:i, :, :] = input[inner_c:i, :, :] * saltarr + saltzero
        inner_c = i

    return input
"""
x_train_ch = salt_and_pepper_custom(x_train)
plt.imshow(np.hstack([x_train_ch[1, :, :], x_train_ch[99, :, :], x_train_ch[101, :, :], x_train_ch[198, :, :]]), cmap='Greys_r')
plt.show()
"""
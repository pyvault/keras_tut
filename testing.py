from dataacq import getset
from keras.datasets import mnist
import numpy as np
import random
(x_train, _), (x_test, _) = mnist.load_data()

print x_train.shape
print type(x_train)
#train_data = getset('data/Train/Posed')
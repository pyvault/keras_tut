from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from keras import callbacks
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
remote = callbacks.RemoteMonitor(root='http://localhost:9000')

input_img = Input(shape=(5200,))
encoded = Dense(4600, activation='relu')(input_img)
encoded = Dense(3200, activation='relu')(encoded)
encoded = Dense(2600, activation='relu')(encoded)

decoded = Dense(3200, activation='relu')(encoded)
decoded = Dense(4600, activation='relu')(decoded)
decoded = Dense(5200, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

data_org = np.load('peporg_np.npy')
data_occ = np.load('pepimgs_np.npy')

train_org=data_org[0:10000,:]
train_occ=data_occ[0:10000,:]
test_org=data_org[10000:12500,:]
test_occ=data_occ[10000:12500,:]
val_org=data_org[12500:15000,:]
val_occ=data_occ[12500:15000,:]

"""
train_org=data_org[0:10,:]
train_occ=data_occ[0:10,:]
test_org=data_org[10:20,:]
test_occ=data_occ[10:20,:]
val_org=data_org[20:30,:]
val_occ=data_occ[20:30,:]
"""

autoencoder.fit(train_occ, train_org,
                nb_epoch=150,
                batch_size=10,
                shuffle=True,
                validation_data=(val_occ, val_org),callbacks=[remote])
autoencoder.save_weights('pep_model.h5')
# encode and decode some digits
# note that we take them from the *test* set

decoded_imgs = autoencoder.predict(test_occ)
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 5  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(test_occ[i].reshape(40, 130))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(40, 130))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
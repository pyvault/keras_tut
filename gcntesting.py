import numpy as np
import matplotlib.pyplot as plt
import csv
from csv_to_npy import csv_convert

#csv_convert('peporg.csv','peporg_np',15000,5200,130,40)
data_train = np.ndarray(shape=(10000, 5200)).astype(np.float32)
data_c = np.load('pepimgs_np.npy')

data_o = np.load('peporg_np.npy')
data_train=data_o[0:10000,:]
#data_train = np.ndarray(shape=(10000, 5200)).astype(np.float32)
#data_train=data_c[0:10000,:]
#print type(data_train)
#print data_train.shape

data_im_o = data_train[0, :]
data_im_c = data_c[0, :]
data_im_ors = np.reshape(data_im_o, (40, 130))
data_im_crs = np.reshape(data_im_c, (40, 130))
data_f=np.hstack([data_im_ors, data_im_crs])
plt.imshow(data_f, cmap='Greys_r')
plt.show()




"""
csv = np.genfromtxt('pepimgs.csv', delimiter=",", dtype=np.float32)
data = np.ndarray(shape=(15000, 5200)).astype(np.float32)
for i in range(1, 15000):
    gcn_im = csv[i, :]

    #print type(gcn_im)
    #print gcn_im.shape
    gcn_im_rs = np.reshape(gcn_im, (130, 40))
    gcn_t=gcn_im_rs.T;
    gcn_t_2=np.reshape(gcn_t,(1,5200))
    #gcn_t_3=np.reshape(gcn_t2,(40,130))
    data[i,:] = gcn_t_2
    #plt.imshow(gcn_im_rs.T, cmap='Greys_r')
    #plt.show()

print type(data)
print data.shape

np.save('pepnp.npy',data)

#data_im = data[1, :]
#data_im_rs = np.reshape(data_im, (40, 130))
#plt.imshow(data_im_rs, cmap='Greys_r')
#plt.show()

"""

"""
import matplotlib.pyplot as plt
from scipy.misc import imsave

eyes = getset('five')
global_contrast_normalize(eyes, scale=55, sqrt_bias=10, use_std=True)

#plt.imshow(gcn_im_rs, cmap='Greys_r')
#plt.show()
for i in range(1, 921):
    gcn_im = eyes[i, :]
    gcn_im_rs = np.reshape(gcn_im, (40, 130))
    string_path = "fivemod/" + str(i) + ".png"
    imsave(string_path, gcn_im_rs)



#print type(eyes_gcn)
#print eyes_gcn.shape
"""
"""from keras.preprocessing.image import ImageDataGenerator


def dirflow(input):
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(input, target_size=(102400, 0))
    return generator
"""

from PIL import Image
import numpy
import glob


def getset(input):
    imagePath = glob.glob(input+'/*.JPG')
    im_array = numpy.array( [numpy.array(Image.open(imagePath[i]).convert('L'), 'f') for i in range(len(imagePath))] )
    flatten_array = im_array.reshape((im_array.shape[0], -1))
    flatten_f_array =flatten_array.astype('float32') / 255.
    print flatten_f_array.shape
    print type(flatten_f_array)
    return flatten_f_array

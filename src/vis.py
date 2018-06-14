import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import load_model 
import cv2


#########################
# utils function
#########################
def tensor_summary(tensor):
    print('shape: {} min: {} max: {}'.format(tensor.shape, tensor.min(), tensor.max()))

def normalize(image):
    image = image.astype(np.float32)
    return (image-image.min())/(image.max()-image.min() + 1e-5)

def display_images(images, titles=None, cols=5, interpolation=None, cmap="gray"):
    # plt.subplots_adjust(wspace=0, hspace=0)
    titles = titles or ['']*len(images)
    rows = math.ceil(len(images)/cols)
    height_ratio = 1.1*(rows/cols)*(0.2 if type(images[0]) is not np.ndarray else 1)
    plt.figure(figsize=(11, 11 * height_ratio))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.axis("off")
        # Is image a list? If so, merge them into one image.
        if type(image) is not np.ndarray:
            image = [normalize(g) for g in image]
            image = np.concatenate(image, axis=1)
        else:
            image = normalize(image)
        plt.title(title, fontsize=9)
        plt.imshow(image, cmap=cmap, interpolation=interpolation)
        i += 1


##############################
# visulization of activations 
##############################

def read_layer(model, x, layer_name):
    get_layer_output = K.function([model.layers[0].input], [model.get_layer(layer_name).output])
    outputs = get_layer_output([x])[0]
    tensor_summary(outputs)
    return outputs[0] 


def view_layer(model, x, layer_name, cols=5):
    outputs = read_layer(model, x, layer_name)
    display_images([outputs[:,:,i] for i in range(10)], cols=cols)


def display_3(model_case, data_case, ind):
    image = data_dict[data_case][ind]
    print(image.shape)
    model = model_dict[model_case]
    display_images([image], cols=2)
    x = image.reshape(1, 45, 45, 1).astype('float32') / 255.
    view_layer(model, x, 'conv2d_1')
    view_layer(model, x, 'conv2d_2')
    view_layer(model, x, 'max_pooling2d_1')


def main():
    data_dict = dict() 
    model_dict = dict()
    case_list = ['raw', 'clean', 'cut']

    for case in case_list:
        data_dict[case] = np.load('../data/mnist_test_{}.npy'.format(case)).reshape(-1, 45, 45)
        model_dict[case] = load_model('../model/cnn_{}.h5'.format(case))

    display_3('cut', 'raw', 1)

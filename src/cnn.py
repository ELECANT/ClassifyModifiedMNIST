import numpy as np
from PIL import Image 

import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K 
from keras.models import load_model 

BATCH_SIZE = 128 
NUM_CLASSES = 10 
EPOCHS = 12 

IMGH, IMGW = 45, 45

FILE_TRAIN_IMG = "../data/mnist_train_raw.npy"
FILE_TRAIN_LABEL = "../data/mnist_train_label"
FILE_TEST_IMG = "../data/mnist_test_raw.npy"
FILE_TEST_LABEL = "../data/mnist_test_label"


def load_data(file_data, file_label):
    data = np.load(file_data)
    label = np.fromfile(file_label, dtype=np.uint8)
    num_data = label.shape[0]
    if K.image_data_format() == 'channels_first':
        data = data.reshape(num_data, 1, IMGH, IMGW)
        input_shape = (1, IMGH, IMGW)
    else:
        data = data.reshape(num_data, IMGH, IMGW, 1)
        input_shape = (IMGH, IMGW, 1)
    data = data.astype('float32') / 255.0
    label = keras.utils.to_categorical(label, NUM_CLASSES)
    return data, label, input_shape 



def build_model(input_shape):
    model = Sequential() 
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(), 
                  metrics=['accuracy'])
    return model 



def train(model_save):
    x_train, y_train, shape_train = load_data(FILE_TRAIN_IMG, FILE_TRAIN_LABEL)
    x_test, y_test, _ = load_data(FILE_TEST_IMG, FILE_TEST_LABEL)
    model = build_model(shape_train)
    
    model.fit(x_train, y_train, 
              batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, 
              validation_data=(x_test, y_test))
    model.save(model_save)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    print('model saved: ', model_save)


# cross evaluation function 
def cross_test():
    test_case = ['raw', 'clean', 'cut']
    for case1 in test_case:
        model_name = '../model/cnn_{}.h5'.format(case1)
        model = load_model(model_name)
        model.summary()
        for case2 in test_case:
            test_data = '../data/mnist_test_{}.npy'.format(case2)
            x_test, y_test, shape_test = load_data(test_data, FILE_TEST_LABEL)
            score = model.evaluate(x_test, y_test, verbose=0)
            print('model: {}, data: {}, loss: {}, accuracy: {}'.format(case1, case2, score[0], score[1]))

if __name__ == '__main__':
    # train('../model/cnn_clean.h5')
    cross_test() 
    
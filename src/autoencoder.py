import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Model
from keras import regularizers

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt


batch_size = 128
num_classes = 10
epochs = 30

# input image dimensions
img_rows, img_cols = 45, 45

FILE_TRAIN_IMG = "../data/mnist_train_raw.npy"
FILE_TRAIN_LABEL = "../data/mnist_train_label"
FILE_TEST_IMG = "../data/mnist_test_raw.npy"
FILE_TEST_LABEL = "../data/mnist_test_label"


x_train, x_test = np.load(FILE_TRAIN_IMG), np.load(FILE_TEST_IMG)
y_train = np.fromfile(FILE_TRAIN_LABEL, dtype=np.uint8)
y_test = np.fromfile(FILE_TEST_LABEL, dtype=np.uint8)

y_test_real=y_test

x_train = x_train.reshape(60000, 45*45)
x_test = x_test.reshape(10000, 45*45)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(45*45,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# this is the size of our encoded representations
encoding_dim = 32 

# this is our input placeholder
input_img = Input(shape=(45*45,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(45*45, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

history = autoencoder.fit(x_train, x_train,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs_train = encoder.predict(x_train)
encoded_imgs_test = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs_test)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(45, 45))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(45, 45))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




encoded_imgs_train_normalized = encoded_imgs_train / np.max(encoded_imgs_train)
encoded_imgs_test_normalized = encoded_imgs_test / np.max(encoded_imgs_test)

# colors = ['r', 'b', 'y', 'm', 'c', 'g', 'k', 'tan', 'orange', 'peru']
# test_colors = [colors[i] for i in y_test_real]
# plt.scatter(encoded_imgs_test_normalized[:, 0], encoded_imgs_test_normalized[:, 1], c=test_colors)
# plt.savefig('2d.png', bbox_inches='tight')

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(encoding_dim,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(encoded_imgs_train_normalized, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(encoded_imgs_test_normalized, y_test))
score = model.evaluate(encoded_imgs_test_normalized, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, np.argmax(y_train, axis=1))
score = logisticRegr.score(x_test, np.argmax(y_test, axis=1))
print(score)

logisticRegr = LogisticRegression()
logisticRegr.fit(encoded_imgs_train_normalized, np.argmax(y_train, axis=1))
score = logisticRegr.score(encoded_imgs_test_normalized, np.argmax(y_test, axis=1))
print(score)


clf_svm = LinearSVC()
clf_svm.fit(x_train, np.argmax(y_train, axis=1))
y_pred_svm = clf_svm.predict(x_test)
acc_svm = accuracy_score(np.argmax(y_test, axis=1), y_pred_svm)
print ("Linear SVM accuracy: ",acc_svm)

clf_svm = LinearSVC()
clf_svm.fit(encoded_imgs_train_normalized, np.argmax(y_train, axis=1))
y_pred_svm = clf_svm.predict(encoded_imgs_test_normalized)
acc_svm = accuracy_score(np.argmax(y_test, axis=1), y_pred_svm)
print ("Linear SVM accuracy: ",acc_svm)


clf_rf = RandomForestClassifier()
clf_rf.fit(x_train, np.argmax(y_train, axis=1))
y_pred_rf = clf_rf.predict(x_test)
acc_rf = accuracy_score(np.argmax(y_test, axis=1), y_pred_rf)
print ("random forest accuracy: ",acc_rf)

clf_rf = RandomForestClassifier()
clf_rf.fit(encoded_imgs_train_normalized, np.argmax(y_train, axis=1))
y_pred_rf = clf_rf.predict(encoded_imgs_test_normalized)
acc_rf = accuracy_score(np.argmax(y_test, axis=1), y_pred_rf)
print ("random forest accuracy: ",acc_rf)

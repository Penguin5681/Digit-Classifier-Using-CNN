import numpy as np
import cv2
import os

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
from keras.preprocessing.image import ImageDataGenerator
from keras.src.utils.np_utils import to_categorical
import pickle

# Config
path = 'myData'

testRatio = 0.2  # 8:2 Training:Testing Ratio
validationRatio = 0.2
imageDimensions = (32, 32, 3)

batchSizeVal = 1
epochsVal = 10
stepsPerEpoch = 2000

images = []
classNo = []
#

myList = os.listdir(path)
print(myList)

classCount = len(myList)

for i in range(0, classCount):
    imageList = os.listdir(path + "/" + str(i))
    for j in imageList:
        currentImage = cv2.imread(path + "/" + str(i) + "/" + j)
        currentImage = cv2.resize(currentImage, (imageDimensions[0], imageDimensions[1]))
        images.append(currentImage)
        classNo.append(i)
    print(i, end=" ")
print(" ")

images = np.array(images)  # (10160, 32, 32, 3)
classNo = np.array(classNo)  # (10160, 32, 32, 3)

# splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validationRatio)
#
numOfSamples = []

for i in range(0, classCount):
    numOfSamples.append(len(np.where(Y_train == i)[0]))

print(numOfSamples)

plot.figure(figsize=(10, 5))
plot.bar(range(0, classCount), numOfSamples)
plot.title("No. of images for each class")
plot.xlabel("Class Id")
plot.ylabel("No. of images")
plot.show()

# preprocessing the image
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


X_train = np.array(list(map(preProcess, X_train)))
X_test = np.array(list(map(preProcess, X_test)))
X_validation = np.array(list(map(preProcess, X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# Augmentation
dataGenerator = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   shear_range=0.1,
                                   rotation_range=10)

dataGenerator.fit(X_train)

# One Hot Encoding
Y_train = to_categorical(Y_train, classCount)
Y_validation = to_categorical(Y_validation, classCount)
Y_test = to_categorical(Y_test, classCount)

def define_model():
    filterCount = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    nodeCount = 500
    model = Sequential()
    model.add((Conv2D(filterCount, sizeOfFilter1, input_shape=(32, 32, 1), activation='relu', kernel_initializer='he_uniform')))
    model.add((Conv2D(filterCount, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(filterCount // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(filterCount // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(nodeCount, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classCount, activation='softmax', kernel_initializer='he_uniform'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = define_model()
print(model.summary())

history = model.fit(dataGenerator.flow(X_train, Y_train, batch_size=batchSizeVal), epochs=epochsVal,
                    steps_per_epoch=stepsPerEpoch,
                    validation_data=(X_validation, Y_validation),
                    shuffle=1)


plot.figure(1)
plot.plot(history.history['loss'])
plot.plot(history.history['val_loss'])
plot.legend(['training', 'validation'])
plot.title('Loss')
plot.xlabel('Epoch')

plot.figure(2)
plot.plot(history.history['accuracy'])
plot.plot(history.history['val_accuracy'])
plot.legend(['training', 'validation'])
plot.title('Accuracy')
plot.xlabel('Epoch')
plot.show()

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test Score =', score[0])
print('Test Accuracy =', score[1])

pickle_output = open('trained_model_data.p', "wb")
pickle.dump(model, pickle_output)
pickle_output.close()

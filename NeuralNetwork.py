from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split

import os
# Number of classes ( 5 detectable objects). Change this in case of different number of classes.
NumClasses = 5
# Dictionary of Classes and their labels. Extend or shrink depending on the number of classes.
CLASSES = {0: "Apple", 1: "Banana", 2: "Book", 3: "Cup", 4: "Ladder"}

# number of samples to take in each class. Extend or reduce depending on your system.
N = 10000

# Number of times to repeat each fruit. Change this depending on your system.
NumEPOCHS = 40

# List of files to load (in order of the dictionary). Extend or shrink depending on the number of classes.
files = ["apple", "banana", "book", "cup", "ladder"]

"""
    Description: This function is used limit the number of samples in each class
                 to N ( Ex: N = 5000, first 5000 elements from each class ).
    Parameters:  array of classes and Number of samples from each class
    Return:      array of classes with limited number of samples
"""


def set_limit(arrays, n):
    # Limit elements from each array up to n elements and return a single list
    new = []
    for array in arrays:
        i = 0
        for item in array:
            if i == n:
                break
            new.append(item)
            i += 1
    return new


"""
    Description: This function is used train, evaluate and the Convolutional 
                 Neural Network model.
    Parameters:  None
    Return:      None
"""


def CNN():
    print("TRAINING CONVOLUTIONAL NEURAL NETWORK !")
    # Load the datasets, The directory is "data/" and the files are
    # reshaped to 28x28 format for the convolutional layers.
    classes = []
    for file in files:
        f = np.load("data/" + file + ".npy")
        new_file = []
        for i in range(len(f)):
            x = np.reshape(f[i], (28, 28))
            x = np.expand_dims(x, axis=0)
            x = np.reshape(f[i], (28, 28, 1))
            new_file.append(x)
        f = new_file
        classes.append(f)
    # limit no of samples in each class to N
    classes = set_limit(classes, N)
    # normalize the values Each of the grayscale images in classes has pixel values between 0 and 255.
    # We want to normalize the values to -1 and 1.
    classes = np.interp(classes, [0, 255], [-1, 1])
    # Make labels for the number of classes and assign them label; 0, 1, 2, 3
    # we extend this to N so first 5000 is 0 next 5000 is 1 and so on
    # "make labels from 0 to NumClasses, each repeated N times"
    labels = []
    for i in range(NumClasses):
        labels += [i] * N

    # prepare data for training and testing, x_train is the images required for training and
    # x_test is the images required for testing. y_train contains corressponding labels for the images in x_train
    # and y_test contains corressponding labels for the images in x_test. 5% is reserved for testing purpose.
    x_train, x_test, y_train, y_test = train_test_split(
        classes, labels, test_size=0.05)

    # Taking standard list of labels and converting it to Matrix form
    Y_train = np_utils.to_categorical(y_train, NumClasses)
    # Y_test = np_utils.to_categorical(y_test, NumClasses)
    ####################Architecture of CNN ###############################
    model = Sequential()
    # Convolutional layer, takes input image and apply filter on top and makes the
    # image more simplified and the computer can understand it better.
    # Defining input shape as 28x28 with 1 color channel (grayscale)
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=(28, 28, 1)))
    # Activation function is relu, which is the rectified linear unit.
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Max pooling layer, takes the max value from the filter and reduces the size of the image (resize layer)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Randomly ignore certain neurons, helps in reducing overfitting
    model.add(Dropout(0.25))
    # Flatten the image to 1D array
    model.add(Flatten())
    # Fully connected layer, takes the input from the previous layer and makes it a single layer
    model.add(Dense(128, activation='relu'))
    # Randomly ignore certain neurons, helps in reducing overfitting
    model.add(Dropout(0.5))
    # We want the number of nodes at the end of neural network to be same as the number of classes
    model.add(Dense(NumClasses, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # training Neural Network on x_train and tell it that x_train features
    # coressponds to y_train labels and batch_size is the number of images to be
    # processed at a time.
    model.fit(np.array(x_train), np.array(Y_train),
              batch_size=32, epochs=NumEPOCHS)

    print("Training complete")

    print("Evaluating model")
    # Evaluate the model on test data, preds is list
    preds = model.predict(np.array(x_test))

    # compare each element of y_test and preds and find the accuracy
    score = 0
    for i in range(len(preds)):
        if np.argmax(preds[i]) == y_test[i]:
            score += 1

    print("Accuracy: ", ((score + 0.0) / len(preds)) * 100)

    # save the model
    model.save("models/CNN" + ".h5")
    print("Model saved")


def MLP():
    print("TRAINING MULTI-LAYER PERCEPTRON !")
    # Load the datasets, The directory is data/ and NPY files are
    # flattened meaning its a 1D array containing 784 elements
    # so we don't need to reshape it.
    classes = []
    for file in files:
        f = np.load("data/" + file + ".npy")
        classes.append(f)
    # limit no of samples in each class to N
    classes = set_limit(classes, N)
    # normalize the values Each of the grayscale images in classes has pixel values between 0 and 255.
    # We want to normalize the values to -1 and 1.
    classes = np.interp(classes, [0, 255], [-1, 1])
    # Make labels for the number of classes and assign them label; 0, 1, 2, 3
    # we extend this to N so first 5000 is 0 next 5000 is 1 and so on
    # "make labels from 0 to NumClasses, each repeated N times"
    labels = []
    for i in range(NumClasses):
        labels += [i] * N
    # prepare data for training and testing, x_train is the images required for training and
    # x_test is the images required for testing. y_train contains corressponding labels for the images in x_train
    # and y_test contains corressponding labels for the images in x_test. 5% is reserved for testing purpose.
    x_train, x_test, y_train, y_test = train_test_split(
        classes, labels, test_size=0.05)

    # Taking standard list of labels and converting it to Matrix form
    Y_train = np_utils.to_categorical(y_train, NumClasses)
    # Y_test = np_utils.to_categorical(y_test, NumClasses)

    ####################Architecture of MLP ###############################
    model = Sequential()
    # First Layer has 600 neurons,with input_dimensions of 784 as we pass
    # the flattened images to the first layer. The activation function is relu.
    model.add(Dense(units=600, activation='relu', input_dim=784))
    model.add(Dropout(0.3))
    # Second Layer has 400 neurons,with input_dimensions of 600 as we pass
    model.add(Dense(units=400, activation='relu'))
    model.add(Dropout(0.3))
    # Third Layer has 200 neurons,with input_dimensions of 400 as we pass
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=25, activation='relu'))
    model.add(Dropout(0.3))
    # Final Layer has 4 neurons
    model.add(Dense(units=NumClasses, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # training Neural Network on x_train and tell it that x_train features
    # coressponds to y_train labels and batch_size is the number of images to be
    # processed at a time.
    model.fit(np.array(x_train), np.array(Y_train),
              batch_size=32, epochs=NumEPOCHS)

    print("Training complete")

    print("Evaluating model")
    # Evaluate the model on test data, preds is list
    preds = model.predict(np.array(x_test))

    # compare each element of y_test and preds and find the accuracy
    score = 0
    for i in range(len(preds)):
        if np.argmax(preds[i]) == y_test[i]:
            score += 1

    print("Accuracy: ", ((score + 0.0) / len(preds)) * 100)

    # save the model
    model.save("models/MLP" + ".h5")
    print("Model saved")


"""
    Description: This function runs the CNN or the MLP model depending on the
                 user input.
    Parameters: Integer representing choice of model.
    Returns:    None 
"""


def execute(argv):
    # Check if user requests CNN model
    if argv == 0:
        # If it already exists, then exit out of the function
        if os.path.isfile("models/CNN.h5"):
            return
        else:
            CNN()
    # Check if user requests MLP
    elif argv == 1:
        # If it already exists, then exit out of the function
        if os.path.isfile("models/MLP.h5"):
            return
        # It doesn't exist, so train the CNN model
        else:
            MLP()

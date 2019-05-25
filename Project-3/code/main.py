# coding: utf-8
from keras.utils import np_utils
import pickle
import gzip
from PIL import Image
import os
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
import PIL.ImageOps

import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.models import load_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

np.set_printoptions(threshold=np.nan)

def get_usps_data_and_edit(path):

    USPSMat = []
    USPSTar = []
    curPath = path
    for j in range(0, 10):
        curFolderPath = curPath + '/' + str(j)
        imgs = os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg, 'r')
                img = img.resize((22, 22))
                new_im = Image.new("L", (28, 28), color=1)
                new_im = PIL.ImageOps.invert(new_im)
                new_im.paste(img, (3, 3))
                imgdata = (255 - np.array(new_im.getdata())) / 255
                USPSMat.append(imgdata)
                USPSTar.append(j)
    USPSVar = np.insert(USPSMat, len(USPSMat[0]), USPSTar, axis=1)
    np.random.shuffle(USPSVar)
    USPSTar = USPSVar[:, [-1]]
    USPSMat = np.delete(USPSVar, np.s_[-1], axis=1)
    return USPSMat, USPSTar

def get_mnist_dataset(path):

    print("Getting MNIST dataset")
    filename = path
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return training_data, validation_data, test_data

def get_usps_data_from_folder(path):
    USPSMat = []
    USPSTar = []
    curPath = path
    for j in range(0, 10):
        curFolderPath = curPath + '/' + str(j)
        imgs = os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg, 'r')
                img = img.resize((28, 28))
                imgdata = (255 - np.array(img.getdata())) / 255
                USPSMat.append(imgdata)
                USPSTar.append(j)
    USPSVar = np.insert(USPSMat, len(USPSMat[0]), USPSTar, axis=1)
    np.random.shuffle(USPSVar)
    USPSTar = USPSVar[:, [-1]]
    USPSMat = np.delete(USPSVar, np.s_[-1], axis=1)

    return USPSMat, USPSTar

def get_usps_dataset(testpath, trainpath,datatype, validation = 0.1):

    if(datatype == "noedit"):
        print("Getting USPS entire dataset")
        USPSMat, USPSTar = get_usps_data_from_folder(testpath)
        USPS_test_data =  (USPSMat , USPSTar)

        USPSMat, USPSTar = get_usps_data_from_folder(trainpath)

        train_end = math.ceil(len(USPSMat)*(1-validation))

        USPS_train_Mat = USPSMat[:train_end]
        USPS_train_Tar = USPSTar[:train_end]
        USPS_train_data = (USPS_train_Mat, USPS_train_Tar)

        USPS_Val_Mat = USPSMat[train_end:]
        USPS_Val_Tar = USPSTar[train_end:]
        USPS_Val_data = (USPS_Val_Mat, USPS_Val_Tar)

        return USPS_train_data, USPS_Val_data, USPS_test_data

    if(datatype == "edit"):
        print("Getting USPS entire dataset and editing it as required")
        USPSMat, USPSTar = get_usps_data_and_edit(testpath)
        USPS_test_data = (USPSMat, USPSTar)

        USPSMat, USPSTar = get_usps_data_and_edit(trainpath)

        train_end = math.ceil(len(USPSMat) * (1 - validation))

        USPS_train_Mat = USPSMat[:train_end]
        USPS_train_Tar = USPSTar[:train_end]
        USPS_train_data = (USPS_train_Mat, USPS_train_Tar)

        USPS_Val_Mat = USPSMat[train_end:]
        USPS_Val_Tar = USPSTar[train_end:]
        USPS_Val_data = (USPS_Val_Mat, USPS_Val_Tar)

        return USPS_train_data, USPS_Val_data, USPS_test_data

def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.array(predictions)
    N = predictions.shape[0]
    true_predictions = np.zeros(shape=np.shape(predictions))
    count = 0
    for i in range(N):
        index = np.argmax(predictions[i])
        true_predictions[i][index] = 0
        if targets[i][index] == 1:
            count += 1
    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    ce = -np.sum(targets*np.log(predictions))/N
    accuracy = (float((count*100))/float(N))

    return [accuracy,ce]


def softmax(pred):
    y = []
    for x in pred:
        y.append(np.exp(x) / np.sum(np.exp(x), axis=0))
    return y

def train_model(data_train, data_val, num_epochs, num_batches, Lambda, learningRate, valbatches):
    epoch = 0

    epochTrainCR = []
    epochTrainAccuracy = []
    epochValCR = []
    epochValAccuracy = []

    X = data_train[0]
    Y = data_train[1]
    Y = np_utils.to_categorical(Y,10)

    bias = np.transpose(np.ones([len(X)]))
    X = np.insert(X, 0, bias, axis=1)


    X_val = data_val[0]
    Y_val = data_val[1]
    Y_val = np_utils.to_categorical(Y_val, 10)

    bias = np.transpose(np.ones([len(X_val)]))
    X_val = np.insert(X_val, 0, bias, axis=1)

    val_datapoints = len(X_val)
    val_batch_size = math.ceil(val_datapoints / valbatches)

    datapoints = len(X)
    batch_size = math.ceil(datapoints/num_batches)
    W_now = np.random.rand(len(X[0]), len(Y[0]))

    print("The batch size is:" + str(batch_size))
    print("Regularization constant = " + str(Lambda))
    print("Learning rate for this epoch is:" + str(learningRate))
    print("Validation set batch size is: " + str(val_batch_size))
    print("Total datapoints in validation set are: " + str(val_datapoints))
    print("Total datapoints in the training set are: " + str(datapoints))
    print("--------------------------------------------------------------")

    while epoch < num_epochs:

        print("--------------------------------------------------------------")
        print("------------------    Epoch : " + str(epoch+1) + "   ----------------------------")
        print("--------------------------------------------------------------")


        next_batch = 0
        current_batch = 0
        batch_number = 1

        next_batch_val = 0
        current_batch_val = 0

        CrossEntropyTrain = []
        AccuracyTrain = []

        CrossEntropyVal = []
        AccuracyVal = []

        while current_batch < datapoints:

            next_batch = current_batch + batch_size
            X_b = X[current_batch:next_batch]
            Y_b = Y[current_batch:next_batch]

            current_batch = next_batch

            next_batch_val = current_batch_val + val_batch_size
            X_val_b = X_val[current_batch_val:next_batch_val]
            Y_val_b = Y_val[current_batch_val:next_batch_val]

            if(next_batch_val >= val_datapoints):
                next_batch_val = 0

            current_batch_val = next_batch_val

            prediction = np.dot(X_b, W_now)
            prediction = softmax(prediction)
            Delta_E_D = -np.dot(np.transpose(X_b) , np.subtract(Y_b,prediction))
            Delta_E_D = np.divide(Delta_E_D , batch_size)

            La_Delta_E_W = np.multiply(Lambda, W_now)

            Delta_E = np.add(Delta_E_D, La_Delta_E_W)

            Delta_W = -np.dot(learningRate, Delta_E)
            W_next = W_now + Delta_W
            W_now = W_next

            # print("The batch number is :"  + str(batch_number) + " Epoch Number is :" + str(epoch))
            batch_number += 1

            prediction = np.dot(X_b, W_now)
            prediction = softmax(prediction)
            CrossEntropyAccuracyTrain = cross_entropy(prediction, Y_b ,)
            CrossEntropyTrain.append(float(CrossEntropyAccuracyTrain[1]))
            AccuracyTrain.append(float(CrossEntropyAccuracyTrain[0]))
            # print("Train Accuracy :" + str(CrossEntropyAccuracyTrain[0])
            #       + " ,Train CrossEntropy:" + str(CrossEntropyAccuracyTrain[1]))


            prediction_val = np.dot(X_val_b, W_now)
            prediction_val = softmax(prediction_val)

            CrossEntropyAccuracyVal = cross_entropy(prediction_val, Y_val_b)
            CrossEntropyVal.append(float(CrossEntropyAccuracyVal[1]))
            AccuracyVal.append(float(CrossEntropyAccuracyVal[0]))
            # print("Validation Accuracy :" + str(CrossEntropyAccuracyVal[0])
            #       + " ,Validation CrossEntropy:" + str(CrossEntropyAccuracyVal[1]))
            # print("----------------------------------------------------------------------")
        epoch += 1
        batchCrossEntropyTrain = np.average(CrossEntropyTrain)
        batchAccuracyTrain = np.average(AccuracyTrain)
        batchCrossEntropyVal = np.average(CrossEntropyVal)
        batchAccuracyVal = np.average(AccuracyVal)
        print("Train CrossEntropy :" + str(batchCrossEntropyTrain)
              + " ,Train Accuracy:" + str(batchAccuracyTrain))
        print("Validation CrossEntropy :" + str(batchCrossEntropyVal)
                  + " ,Validation Accuracy:" + str(batchAccuracyVal))

        epochTrainAccuracy.append(batchAccuracyTrain)
        epochTrainCR.append(batchCrossEntropyTrain)
        epochValCR.append(batchCrossEntropyVal)
        epochValAccuracy.append(batchAccuracyVal)
    plotting(epochTrainAccuracy, epochTrainCR, epochValAccuracy, epochValCR)
    return W_now

def plotting(TrainAcc, TrainCR, ValAcc, ValCR):
    plt.subplot(2, 2, 1)
    plt.plot(TrainCR)
    plt.title('Training CrossEntropy')
    plt.xlabel('Number of epochs')

    plt.subplot(2, 2, 2)
    plt.plot(ValCR)
    plt.title("Validation CrossEntropy")
    plt.xlabel('Number of epochs')

    plt.subplot(2, 2, 3)
    plt.plot(TrainAcc)
    plt.title('Training Accuracy')
    plt.xlabel('Number of epochs')

    plt.subplot(2, 2, 4)
    plt.plot(ValAcc)
    plt.title("Validation Accuracy")
    plt.xlabel('Number of epochs')
    plt.show()

def test_model(data_test, f_name):
    X = data_test[0]
    Y = data_test[1]
    Y = np_utils.to_categorical(Y,10)
    weights = np.genfromtxt(f_name, delimiter=',')
    bias = np.transpose(np.ones([len(X)]))
    X = np.insert(X, 0, bias, axis=1)

    prediction = np.dot(X, weights)
    prediction = softmax(prediction)

    prediction = np.array(prediction)
    N = prediction.shape[0]

    true_predictions = np.zeros(shape=(len(prediction),))
    true_target = np.zeros(shape=(len(Y),))
    count = 0
    for i in range(N):
        index = np.argmax(prediction[i])
        true_predictions[i] = index

        index1 = np.argmax(Y[i])
        true_target[i] = index1
        if Y[i][index] == 1:
            count += 1

    create_confusion_matrix(true_predictions, true_target)


    return cross_entropy(prediction, Y, ), true_predictions

def train_cnn_nn(trainingData, trainingLabel, validationData, validationLabel, train_name):
    batch_size = 64
    num_classes = 10
    epochs = 20
    input_shape = (28, 28, 1)
    trainingData = trainingData.reshape(trainingData.shape[0], 28, 28, 1)
    validationData = validationData.reshape(validationData.shape[0], 28, 28, 1)
    trainingLabel = np_utils.to_categorical(trainingLabel, 10)
    validationLabel = np_utils.to_categorical(validationLabel, 10)

    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.0001)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    tensorboard_cb_cnn = TensorBoard(log_dir='logs', batch_size=32, write_graph=True)

    datagen.fit(trainingData)
    h = model.fit_generator(datagen.flow(trainingData, trainingLabel, batch_size=batch_size),
                            epochs=epochs, validation_data=(validationData, validationLabel),
                            verbose=True, steps_per_epoch=trainingData.shape[0] // batch_size
                            , callbacks=[learning_rate_reduction,tensorboard_cb_cnn], )


    f_name = train_name + "cnnmodel.h5"
    model.save(f_name)  # creates a HDF5 file 'my_model.h5'
    df = pd.DataFrame(h.history)
    df.plot(subplots=True, grid=True, figsize=(5, 7))
    plt.show()

def test_cnn(testingData, testingLabel, train_name, test_name):
    f_name = train_name + "cnnmodel.h5"
    model = load_model(f_name)
    testingData = testingData.reshape(testingData.shape[0], 28, 28, 1)
    testingLabel_one_hot = np_utils.to_categorical(testingLabel, 10)


    true_predictions = model.predict_classes(testingData)
    score = model.evaluate(testingData, testingLabel_one_hot, verbose=False)
    print('Test CrossEntropy using Convolutional Neural Networks on dataset ' + test_name + " after training on " + train_name +' is: ', score[0])
    print('Test Accuracy using Convolutional Neural Networks on dataset ' + test_name + " after training on " + train_name +' is: ', score[1])
    create_confusion_matrix(true_predictions, testingLabel)

    return  true_predictions

def get_model(num_hidden_nodes, image_size, num_classes, hidden_activation):
    model = Sequential()
    model.add(Dense(units=num_hidden_nodes, activation=hidden_activation, input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_nn(trainingData, trainingLabel, validationData, validationLabel, train_name, num_epochs, batch_size):

    num_classes = 10
    num_epochs = num_epochs
    model_batch_size = batch_size
    tb_batch_size = batch_size
    early_patience = 20
    trainingLabel = np_utils.to_categorical(trainingLabel, 10)
    validationLabel = np_utils.to_categorical(validationLabel, 10)

    image_size = 784
    model = get_model(num_hidden_nodes=512, image_size = image_size, num_classes=num_classes, hidden_activation='relu')

    tensorboard_cb = TensorBoard(log_dir='logs', batch_size=tb_batch_size, write_graph=True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')


    history = model.fit(trainingData
                        , trainingLabel
                        , validation_data=(validationData, validationLabel)
                        , epochs=num_epochs
                        , batch_size=model_batch_size
                        , callbacks=[tensorboard_cb, earlystopping_cb]
                        )
    f_name = train_name + "nnmodel.h5"
    model.save(f_name)  # creates a HDF5 file 'my_model.h5'
    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(5, 7))
    plt.show()

def test_nn(testingData, testingLabel, train_name, test_name):
    f_name = train_name + "nnmodel.h5"
    model = load_model(f_name)
    testingLabel_one_hot = np_utils.to_categorical(testingLabel, 10)

    true_predictions = model.predict_classes(testingData)
    score = model.evaluate(testingData, testingLabel_one_hot, verbose=False)
    print('Test CrossEntropy using Neural Networks on dataset ' + test_name + " trained on " + train_name + ' is:', score[0])
    print('Test Accuracy using Neural Networks on dataset '+ test_name + " trained on " + train_name + ' is:', score[1])
    create_confusion_matrix(true_predictions, testingLabel)

    return true_predictions


def read_data():

    mnist_train, mnist_val, mnist_test = get_mnist_dataset('mnist.pkl.gz')
    usps_train, usps_val, usps_test = get_usps_dataset('USPSdata/Test', 'USPSdata/Numerals',  datatype = "noedit", validation = 0.1)

    return mnist_train, mnist_val, mnist_test, usps_train, usps_val, usps_test

def train_logistic(train, val, train_set, num_epochs, num_batches, Lambda, learningRate, valbatches):
    weights = train_model(train, val, num_epochs, num_batches, Lambda, learningRate, valbatches)
    f_name = "LR"+train_set+".csv"
    np.savetxt(f_name, weights, delimiter=",")
    print("-----------------------------------------------------------------------")
    print("The model is trained for " + train_set + " dataset ")
    print("-----------------------------------------------------------------------")

def test_logistic(test1, train_set, test_set):

    f_name = "LR" + train_set + ".csv"
    test1_results, prediction = test_model(test1, f_name)
    print("The Accuracy for " + test_set + " when trained on " + train_set + " dataset are: " + str(test1_results[0]))
    print("The Cross Entropy for " + test_set + " when trained on " + train_set + " dataset are: " + str(test1_results[1]))

    return prediction


def random_forest(x_train, y_train, train_name):

    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x_train, y_train.ravel())
    filename =  train_name + '_model_RF.sav'
    pickle.dump(clf, open(filename, 'wb'))

def random_forest_test(x_test1, y_test1, train_name, test_name):

    filename = train_name + '_model_RF.sav'
    clf = pickle.load(open(filename, 'rb'))
    prediction_validation = clf.predict(x_test1)
    print("When the training has been done on: " + train_name + " and tested on " + test_name )
    print("Testing Accuracy: " + str(accuracy_score(y_test1, prediction_validation)))
    print("Testing Confusion Matrix: \n" + str(confusion_matrix(y_test1, prediction_validation)))

    return prediction_validation


def svm_train(x_train, y_train, train_name):

    # param_C = 5
    # param_gamma = 0.05
    # clf = SVC(verbose = True, C=param_C,gamma=param_gamma)
    clf = SVC(verbose = True, kernal='linear')
    clf.fit(x_train, y_train.ravel())
    filename =  train_name + '_model_svm.sav'
    pickle.dump(clf, open(filename, 'wb'))

def svm_test(x_test1, y_test1, train_name, test_name):

    filename = train_name + '_model_svm.sav'
    clf = pickle.load(open(filename, 'rb'))
    prediction_validation = clf.predict(x_test1)
    print("When the training has been done on: " + train_name + " and tested on " + test_name )
    print("Testing Accuracy: " + str(accuracy_score(y_test1, prediction_validation)))
    print("Testing Confusion Matrix: \n" + str(confusion_matrix(y_test1, prediction_validation)))

    return prediction_validation


def create_confusion_matrix(true_predictions, target):

    print("Testing Confusion Matrix: \n" + str(confusion_matrix(target, true_predictions)))

def ensemble_classifier(svmop, lrop,nnop, rfop, test):
    testop = []
    X = np.concatenate((svmop.reshape(-1,1), lrop.reshape(-1,1)), axis = 1)
    X = np.insert(X, 0, nnop, axis=1)
    X = np.insert(X, 0, rfop, axis=1)
    X = X.astype(int)
    test = test.astype(int)
    for row in X:
        counts = np.bincount(row)
        if(counts[np.argmax(counts)] == 1):
            testop.append(row[0])
        else:
            testop.append(np.argmax(counts))
    print("Testing Accuracy: " + str(accuracy_score(test, testop)))
    print("Testing Confusion Matrix: \n" + str(confusion_matrix(test, testop)))


print("--------------------------------------------------")
print("Reading all the datasets")
print("--------------------------------------------------")

mnist_train, mnist_val, mnist_test, usps_train, usps_val, usps_test = read_data()
# edit_usps_train, edit_usps_val, edit_usps_test = get_usps_dataset('USPSdata/Test', 'USPSdata/Numerals', datatype = "edit", validation = 0.1 )


# print("--------------------------------------------------")
# print("Training all the models on MNIST Dataset")
# print("--------------------------------------------------")
#
train_cnn_nn(mnist_train[0], mnist_train[1], mnist_val[0], mnist_val[1], train_name="mnist")
# train_logistic(mnist_train, mnist_val, "mnist", num_epochs=200, num_batches=500, Lambda=0.0, learningRate=0.01, valbatches=10)
# train_nn(mnist_train[0], mnist_train[1], mnist_val[0], mnist_val[1], "mnist", num_epochs=200, batch_size=100)
# random_forest(mnist_train[0], mnist_train[1], train_name="mnist")
# svm_train(mnist_train[0], mnist_train[1], train_name="mnist")
#
#
# print("--------------------------------------------------")
# print("Training all the models on USPS Dataset")
# print("--------------------------------------------------")
#
# train_cnn_nn(usps_train[0],usps_train[1], usps_val[0],usps_val[1], train_name = "usps")
# train_logistic(usps_train, usps_val, "usps" ,num_epochs=200 ,num_batches=100, Lambda = 0.0, learningRate=0.01, valbatches=2)
# train_nn(usps_train[0],usps_train[1], usps_val[0],usps_val[1], "usps", num_epochs=200, batch_size=100)
# random_forest(usps_train[0], usps_train[1],train_name = "usps")
# svm_train(usps_train[0], usps_train[1],train_name = "usps")
#
#
# print("--------------------------------------------------")
# print("Training all the models on EDIT-USPS Dataset")
# print("--------------------------------------------------")
#
# train_cnn_nn(edit_usps_train[0],edit_usps_train[1], edit_usps_val[0],edit_usps_val[1], train_name = "edit_usps")
# train_logistic(edit_usps_train, edit_usps_val, "edit_usps" ,num_epochs=200 ,num_batches=100, Lambda = 0.0, learningRate=0.01, valbatches=2)
# train_nn(edit_usps_train[0],edit_usps_train[1], edit_usps_val[0],edit_usps_val[1], "edit_usps", num_epochs=200, batch_size=100)
# random_forest(edit_usps_train[0], edit_usps_train[1],train_name = "edit_usps")
# svm_train(edit_usps_train[0], edit_usps_train[1],train_name = "edit_usps")

##################################################################################################################
##################################################################################################################

print("-----------------------------------------------------------------------------------------------")

# When the model has been trained on MNIST dataset and the testing is also on MNIST dataset

cnnoutput = test_cnn(mnist_test[0], mnist_test[1], train_name = "mnist", test_name = "mnist" )
print("-----------------------------------------------------------------------------------------------")
lroutput = test_logistic(mnist_test, "mnist", "mnist")
print("-----------------------------------------------------------------------------------------------")
nnoutput = test_nn(mnist_test[0], mnist_test[1], train_name = "mnist", test_name = "mnist" )
print("-----------------------------------------------------------------------------------------------")
rfoutput = random_forest_test(mnist_test[0], mnist_test[1], train_name = "mnist", test_name = "mnist" )
print("-----------------------------------------------------------------------------------------------")
svmoutput = svm_test(mnist_test[0], mnist_test[1], train_name = "mnist", test_name = "mnist" )
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")
ensemble_classifier(svmoutput, lroutput, nnoutput, rfoutput, mnist_test[1])
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")

# When the model has been trained on MNIST and the testing is done on USPS

cnnoutput = test_cnn(usps_test[0], usps_test[1],train_name = "mnist", test_name = "usps" )
print("-----------------------------------------------------------------------------------------------")
lroutput = test_logistic(usps_test, "mnist", "usps" )
print("-----------------------------------------------------------------------------------------------")
nnoutput = test_nn(usps_test[0], usps_test[1],train_name = "mnist", test_name = "usps" )
print("-----------------------------------------------------------------------------------------------")
rfoutput = random_forest_test(usps_test[0], usps_test[1],train_name = "mnist", test_name = "usps" )
print("-----------------------------------------------------------------------------------------------")
svmoutput = svm_test(usps_test[0], usps_test[1],train_name = "mnist", test_name = "usps" )
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")
ensemble_classifier(svmoutput, lroutput, nnoutput, rfoutput, usps_test[1])
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")

# When the model has been trained on MNIST and the testing is done on EDIT_USPS

cnnoutput = test_cnn(edit_usps_test[0], edit_usps_test[1], train_name = "mnist",test_name = "edit_usps")
print("-----------------------------------------------------------------------------------------------")
lroutput = test_logistic(edit_usps_test, "mnist", "edit_usps")
print("-----------------------------------------------------------------------------------------------")
nnoutput = test_nn(edit_usps_test[0], edit_usps_test[1], train_name = "mnist",test_name = "edit_usps")
print("-----------------------------------------------------------------------------------------------")
rfoutput = random_forest_test(edit_usps_test[0], edit_usps_test[1], train_name = "mnist",test_name = "edit_usps")
print("-----------------------------------------------------------------------------------------------")
svmoutput = svm_test(edit_usps_test[0], edit_usps_test[1], train_name = "mnist",test_name = "edit_usps")
# print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")

ensemble_classifier(svmoutput, lroutput, nnoutput, rfoutput, edit_usps_test[1])
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")

# When the model has been trained on USPS and tested on USPS

cnnoutput = test_cnn(usps_test[0], usps_test[1], train_name = "usps", test_name="usps")
print("-----------------------------------------------------------------------------------------------")
lroutput = test_logistic(usps_test,"usps", "usps")
print("-----------------------------------------------------------------------------------------------")
nnoutput = test_nn(usps_test[0], usps_test[1], train_name = "usps", test_name="usps")
print("-----------------------------------------------------------------------------------------------")
rfoutput = random_forest_test(usps_test[0], usps_test[1], train_name = "usps", test_name="usps")
print("-----------------------------------------------------------------------------------------------")
svmoutput = svm_test(usps_test[0], usps_test[1], train_name = "usps", test_name="usps")
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")
ensemble_classifier(svmoutput, lroutput, nnoutput, rfoutput, usps_test[1])
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")

# When the model has been trained on USPS and tested on MNIST

cnnoutput = test_cnn(mnist_test[0], mnist_test[1], train_name = "usps", test_name="mnist")
print("-----------------------------------------------------------------------------------------------")
lroutput = test_logistic(mnist_test, "usps", "mnist")
print("-----------------------------------------------------------------------------------------------")
nnoutput = test_nn(mnist_test[0], mnist_test[1], train_name = "usps", test_name="mnist")
print("-----------------------------------------------------------------------------------------------")
rfoutput = random_forest_test(mnist_test[0], mnist_test[1], train_name = "usps", test_name="mnist")
print("-----------------------------------------------------------------------------------------------")
svmoutput = svm_test(mnist_test[0], mnist_test[1], train_name = "usps", test_name="mnist")
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")
ensemble_classifier(svmoutput, lroutput, nnoutput, rfoutput, mnist_test[1])
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")

# When the model has been trained on EDIT-USPS and tested on EDIT-USPS

cnnoutput = test_cnn(edit_usps_test[0],edit_usps_test[1],train_name = "edit_usps", test_name = "edit_usps")
print("-----------------------------------------------------------------------------------------------")
lroutput = test_logistic(edit_usps_test, "edit_usps", "edit_usps")
print("-----------------------------------------------------------------------------------------------")
nnoutput = test_nn(edit_usps_test[0],edit_usps_test[1], train_name = "edit_usps", test_name = "edit_usps")
print("-----------------------------------------------------------------------------------------------")
rfoutput = random_forest_test(edit_usps_test[0], edit_usps_test[1],  train_name = "edit_usps", test_name="edit_usps")
print("-----------------------------------------------------------------------------------------------")
svmoutput = svm_test(edit_usps_test[0], edit_usps_test[1],train_name = "edit_usps", test_name = "edit_usps")
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")
ensemble_classifier(svmoutput, lroutput, nnoutput, rfoutput, edit_usps_test[1])
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")

# When the model has been trained on EDIT-USPS and tested on MNIST

cnnoutput = test_cnn(mnist_test[0],mnist_test[1],train_name = "edit_usps", test_name = "mnist")
print("-----------------------------------------------------------------------------------------------")
lroutput = test_logistic(mnist_test, "edit_usps", "mnist")
print("-----------------------------------------------------------------------------------------------")
nnoutput = test_nn(mnist_test[0],mnist_test[1], train_name = "edit_usps", test_name = "mnist")
print("-----------------------------------------------------------------------------------------------")
rfoutput = random_forest_test(mnist_test[0], mnist_test[1], train_name = "edit_usps", test_name="mnist")
print("-----------------------------------------------------------------------------------------------")
svmoutput = svm_test(mnist_test[0], mnist_test[1],train_name="edit_usps", test_name = "mnist")
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")
ensemble_classifier(svmoutput, lroutput, nnoutput, rfoutput, mnist_test[1])
print("-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------")

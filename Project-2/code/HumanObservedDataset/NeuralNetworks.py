import pandas as pd
from keras.utils import np_utils
from sklearn.cluster import KMeans
import numpy as np
import csv
import math
from matplotlib import pyplot as plt

## Tried how the binary values for inputs work on neural network
## Also tested neural network by preprocessing data by normalization
##  i.e. mean =0 and variance = 1.

def processDatapoint(datamatrix):
    binlist = []
    newDataMatrix = []
    for i in range(len(datamatrix)):
        dataRow = []
        for j in range(len(datamatrix[0])):
            binlist = []
            if datamatrix[i][j] >= 0:
                binform = "1" + bin(datamatrix[i][j])[2:].zfill(3)
            else:
                unsigned = abs(datamatrix[i][j])
                binform = "0" + bin(unsigned)[2:].zfill(3)
            for k in range(len(binform)):
                binlist.append(int(binform[k]))
            dataRow = dataRow + binlist
        newDataMatrix.append(dataRow)
    return newDataMatrix

## Used to load processed csv files and drop the first two coloumns
## The first two coloumns contain only image ID's so that is not used.
## The features are asssigned to datamatrix and targets to labels
def processData(dataset, singular_matrix):
    data = pd.read_csv(dataset)
    labels = data['target'].values
    labels = np_utils.to_categorical(np.array(labels),2)
    data = data.drop(data.columns[[0, 1, -1]], axis=1)
    data = data.drop(data.columns[singular_matrix], axis=1)
    datamatrix = data.as_matrix()
    # print(len(datamatrix))
    # print(len(datamatrix[0]))
    # datamatrix = processDatapoint(datamatrix)
    # print(len(datamatrix))
    # print(len(datamatrix[0]))
    # datamatrix = data.as_matrix()
    return datamatrix, labels

## First all the singular coloums have been deleted as they are irrelevent and do not decide anything
def singular_features(filePath):
    dataMatrix = []
    singular_feature = []
    iterator = 0
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            iterator += 1
            if (iterator > 1):
                dataRow = []
                for column in row:
                    dataRow.append(column)
                dataMatrix.append(dataRow)

    dataMatrix = np.delete(dataMatrix, [0,1], axis=1)
    dataMatrix = np.delete(dataMatrix, np.s_[-1:], axis=1)
    dataMatrix = [[int(float(j)) for j in i] for i in dataMatrix]
    dataMatrix = np.transpose(dataMatrix)
    BigSigma = GenerateBigSigma(dataMatrix)
    for i in range(len(BigSigma)):
        if BigSigma[i][i] == 0:
            singular_feature.append(i)
    return singular_feature

## Variance for all the features has been calculated so that the coloumns with
## zero variance could be deleted.
def GenerateBigSigma(Data):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = len(DataT)
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])
        varVect.append(np.var(vct))

    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]

    return BigSigma

## This function was used for data Normalization. For normalizing the data each feature is
## subtracted with corresponding mean and divided by net variance.


def GenerateMeanVar(Data):
    Variance    = np.zeros(len(Data))
    mean        = np.zeros(len(Data))
    DataT       = np.transpose(Data)
    TrainingLen = len(DataT)
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])
        Variance[i] = np.var(vct)
        mean[i] = np.mean(vct)

    return Variance, mean

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from matplotlib import pyplot
from keras import regularizers


## The data sets are first uploaded over here using the above defined functions.
## The data is presorted according to all three criterias for seen, unseen, and
## shuffled. Each dataset can be seperately uploaded and the results can be observed.
singular_data   = singular_features('training.csv')
trainingData, trainingLabel = processData('training.csv',singular_data)
testingData, testingLabel = processData('testing.csv',singular_data)
validationData, validationLabel = processData('validation.csv',singular_data)

input_size = len(trainingData[0])
first_dense_layer_nodes = 128
second_dense_layer_nodes = 64
drop_out = 0.1
last_dense_layer_nodes  = len(validationLabel[0])

## A two layer neural network model has been defined which is sed for classification
## Here we have all the features as input and at the last layer we use softmax to predict the
## actual output.
def get_model():
    model = Sequential()
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(last_dense_layer_nodes))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = get_model()
num_epochs = 1000
model_batch_size = 100
tb_batch_size = 100
early_patience = 100

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')


history = model.fit(trainingData
                    , trainingLabel
                    , validation_data=(validationData,validationLabel)
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )

score = model.evaluate(testingData, testingLabel, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(5,7))
pyplot.show()

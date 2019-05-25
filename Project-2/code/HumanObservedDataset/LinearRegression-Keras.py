import pandas as pd
from keras.utils import np_utils
from sklearn.cluster import KMeans
import numpy as np
import csv
import math
from matplotlib import pyplot as plt

## We have processed the data set entirely i.e. calculated the K means center and
## deleted the features with singuar values and also returned the final phi matrix and
## also returned the final target values
def processData(dataset, singular_matrix):
    M = 9
    data = pd.read_csv(dataset)
    labels = data['target'].values
    data = data.drop(data.columns[[0, 1, -1]], axis=1)
    data = data.drop(data.columns[singular_matrix], axis=1)
    datamatrix = data.as_matrix()
    datamatrix = np.transpose(datamatrix)
    # labels = np_utils.to_categorical(np.array(labels), 2)
    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(datamatrix))
    Mu = kmeans.cluster_centers_

    BigSigma = GenerateBigSigma(datamatrix)
    phi_matrix = GetPhiMatrix(datamatrix, Mu, BigSigma, 100)
    return phi_matrix, labels

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
    # print(len(BigSigma))
    #print ("Data Matrix Generated..")
    return singular_feature

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
    #BigSigma = np.dot(1,BigSigma)
    return BigSigma

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 100):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    bias = np.ones((int(len(PHI)), 1))
    PHI = np.append(PHI, bias, 1)
    return PHI

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetScalar(DataRow,MuRow, BigSigInv):
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))
    L = np.dot(R,T)
    return L

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from matplotlib import pyplot

import numpy as np

singular_data   = singular_features('training.csv')
trainingData, trainingLabel = processData('training.csv',singular_data)
testingData, testingLabel = processData('testing.csv',singular_data)
validationData, validationLabel = processData('validation.csv',singular_data)

input_size = len(trainingData[0])
first_dense_layer_nodes  = 1

## The model for logistic regression has been defined.
## The model is a 2 layer NN with only 1 neuron in the output layer
## and 1 input layer as the number of features as the input.
## The SGD has been used as the optimizer and the mean square error has
## been continously monitored.
def get_model():
    model = Sequential()
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.summary()
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

model = get_model()
num_epochs = 15
model_batch_size = 1
tb_batch_size = 1
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

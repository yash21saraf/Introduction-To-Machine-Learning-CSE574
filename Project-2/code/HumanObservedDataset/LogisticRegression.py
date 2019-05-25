# coding: utf-8
import imp
from sklearn.cluster import KMeans
import numpy as np
import csv
import math
from matplotlib import pyplot as plt

## The parameters have been defined
C_Lambda = 1
learningRate = 0.01

## The data has been seerated into target and feature values.
def GetTargetVector(filePath):
    t = []
    iterator = 0
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            iterator += 1
            if (iterator > 1):
                t.append(float(row[-1]))
    return np.array(t)

def GenerateRawData(filePath, singular_data):
    dataMatrix = []
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
    dataMatrix = np.delete(dataMatrix, singular_data, axis=1)
    dataMatrix = np.delete(dataMatrix, np.s_[-1:], axis=1)
    dataMatrix = [[int(float(j)) for j in i] for i in dataMatrix]
    bias = np.ones((int(len(dataMatrix)), 1))
    dataMatrix = np.append(dataMatrix, bias, 1)

    return dataMatrix

# ## This calculates the output using the PHI matrix and the weights obtained from Moore Penrose Inversion.
def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    Y = (1/(1+ np.exp(-Y)))
    ##print ("Test Out Generated..")
    return Y

## Herewe have the loss function as the cross entropy function which has been defined and used below
def GetCrossEntropy(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum - ((1-ValDataAct[i])*np.log(1-VAL_TEST_OUT[i]) + ValDataAct[i]*np.log(VAL_TEST_OUT[i]))
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1

    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    return [str(accuracy),str(sum/len(VAL_TEST_OUT))]

## The covariacne values have been calculated
def GenerateBigSigma(Data):
    BigSigma    = np.zeros((len(Data),len(Data)))
    # print(len(Data))
    DataT       = np.transpose(Data)
    TrainingLen = len(DataT)
    # print(len(DataT[0]))
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])
        varVect.append(np.var(vct))

    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]

    #BigSigma = np.dot(1,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

## The features which have 0 variance has been deleted
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

## The features with 0 variance has been extracted so that they can be deleted as they throw the
## singular matrix error. Also these features are of no use as they are same for all the datapoints
## so they cause no change in the output.
singular_data   = singular_features('training.csv')
print("Singular feature columns are:" + str(singular_data))
print("---------PLEASE WAIT----------------")
print("Generating training target Vector")
trainingTarget = GetTargetVector('training.csv')
print("---------PLEASE WAIT----------------")
print("Generating training Rawdata Vector")
trainingData   = GenerateRawData('training.csv',singular_data)
print("---------PLEASE WAIT----------------")
print("Generating testing target Vector")
testingTarget = GetTargetVector('testing.csv')
print("---------PLEASE WAIT----------------")
print("Generating testing Rawdata Vector")
testingData   = GenerateRawData('testing.csv',singular_data)
print("---------PLEASE WAIT---------------")
print("Generating valiadtion target Vector")
validationTarget = GetTargetVector('validation.csv')
print("---------PLEASE WAIT----------------")
print("Generating validation Rawdata Vector")
validationData   = GenerateRawData('validation.csv',singular_data)
print("All data matrices has been created.")
print ('----------------------------------------------------')
print("Shape of Training feature data" + str(trainingData.shape))
print("Shape of Training Target data" + str(trainingTarget.shape))
print("Shape of Testing feature data" + str(testingData.shape))
print("Shape of Testing Target data" + str(testingTarget.shape))
print("Shape of Validation feature data" + str(validationData.shape))
print("Shape of Validation Target data" + str(validationTarget.shape))
print ('----------------------------------------------------')
print("This might take some time. Please wait")
print("---------------------------------------------------------------------------------------")
print ('UBITname      = ysaraf')
print ('Person Number = 50290453')
print ('----------------------------------------------------')
print ("-----------Handwriting features dataset-------------")
print ('----------------------------------------------------')
print ("-------SGD with Radial Basis Function---------------")
print ('----------------------------------------------------')
print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')

CrossEntropyArr = []
AccuracyArr = []

## Random initialization of weights is done. The extra 1 term has been added for the bias.
W_Now        = np.random.rand(len(trainingData[0]),)
La           = C_Lambda
L_CrossEntropy_Val   = []
L_Accuracy_Val = []
L_CrossEntropy_TR    = []
L_Accuracy_TR    = []
L_CrossEntropy_Test  = []
L_Accuracy_Test  = []
W_Mat        = []
print("These iterations might take some significant time depending upon the number of basis function taken")
for i in range(len(trainingData)):

    print ('---------Iteration: ' + str(i) + '--------------')
    prediction = np.dot(np.transpose(W_Now),trainingData[i])
    prediction = (1/(1+ np.exp(-prediction)))
    Delta_E_D     = -np.dot((trainingTarget[i] - prediction),trainingData[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next

    TR_TEST_OUT   = GetValTest(trainingData,W_T_Next)
    CrossEntropy_TR       = GetCrossEntropy(TR_TEST_OUT,trainingTarget)
    L_CrossEntropy_TR.append(float(CrossEntropy_TR[1]))
    L_Accuracy_TR.append(float(CrossEntropy_TR[0]))
    print("Training Target: " + str(trainingTarget[i]) + " ,Prediction:" + str(prediction))
    print("Train Accuracy :" + str(CrossEntropy_TR[0]) + " ,Train CrossEntropy:" + str(CrossEntropy_TR[1]))

    # -----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(validationData,W_T_Next)
    CrossEntropy_Val      = GetCrossEntropy(VAL_TEST_OUT,validationTarget)
    L_CrossEntropy_Val.append(float(CrossEntropy_Val[1]))
    L_Accuracy_Val.append(float(CrossEntropy_Val[0]))
    print("Val Accuracy :" + str(CrossEntropy_Val[0]) + " ,Val CrossEntropy:" + str(CrossEntropy_Val[1]))

TEST_OUT_GD          = GetValTest(testingData,W_Now)
TestAccuracyGD       = GetCrossEntropy(TEST_OUT_GD,testingTarget)
print ("E_rms Testing    = " + str(TestAccuracyGD[1]))
print ("Testing Accuracy = " + str(TestAccuracyGD[0]))

print ('----------Gradient Descent Solution--------------------')
print ("Learning Rate used: " + str(learningRate))
print ("Regularization constant: " + str(La))
print ("CrossEntropy Training   = " + str(np.around(min(L_CrossEntropy_TR),5)))
print ("CrossEntropy Validation = " + str(np.around(min(L_CrossEntropy_Val),5)))

## Plots for the training and validation sets are made for CrossEntropy and accuracy.
plt.subplot(2, 2, 1)
plt.plot(L_CrossEntropy_TR)
plt.title('Training CrossEntropy')
plt.xlabel('Number of datapoints fed:')

plt.subplot(2, 2, 2)
plt.plot(L_CrossEntropy_Val)
plt.title("Validation CrossEntropy")
plt.xlabel('Number of datapoints fed:')

plt.subplot(2, 2, 3)
plt.plot(L_Accuracy_TR)
plt.title('Training Accuracy')
plt.xlabel('Number of datapoints fed:')

plt.subplot(2, 2, 4)
plt.plot(L_Accuracy_Val)
plt.title("Validation Accuracy")
plt.xlabel('Number of datapoints fed:')

plt.show()

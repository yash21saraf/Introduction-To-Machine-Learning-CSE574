# coding: utf-8

import imp
from sklearn.cluster import KMeans
import numpy as np
import csv
import math
from matplotlib import pyplot as plt
## The params have been defined
C_Lambda = 0.1
learningRate = 0.01
M = 9
PHI = []
## The data has been processed usig the following functions

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

## Here we calculate the data columns with 0 variance and delete them as they have 0 significance.
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
    dataMatrix = np.transpose(dataMatrix)
    return dataMatrix

## Big SIgma is nothing but the covariance matrix
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

## The weights have been calculated using closed form
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    return W

## The design matrix for all the datasets have been created
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

## RBF is nothing but a gaussian function which is calculated
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetScalar(DataRow,MuRow, BigSigInv):
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))
    L = np.dot(R,T)
    return L

# ## This calculates the output using the PHI matrix and the weights obtained from Moore Penrose Inversion.
def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

## For linear regression we have calculated the value of ERMS
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1

    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    return [str(accuracy),str(math.sqrt(sum/len(VAL_TEST_OUT)))]

## Used to delete the coloumns containing varaince as 0.
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
print("Finding out the k means cluster and Phi matrix for all the datasets")
print("Also finding out the closed form solution.")
print("This might take some time. Please wait")
ErmsArr = []
AccuracyArr = []

kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(trainingData))
Mu = kmeans.cluster_centers_

BigSigma     = GenerateBigSigma(trainingData)
TRAINING_PHI = GetPhiMatrix(trainingData, Mu, BigSigma, 100)
TEST_PHI     = GetPhiMatrix(testingData, Mu, BigSigma, 100)
VAL_PHI      = GetPhiMatrix(validationData, Mu, BigSigma, 100)

W            = GetWeightsClosedForm(TRAINING_PHI,trainingTarget,(C_Lambda))

print("Shape of feature center matrix" + str(Mu.shape))
print("Shape of Variance matrix" + str(BigSigma.shape))
print("Training design matrix" + str(TRAINING_PHI.shape))
print("Shape of weights calculated" + str(W.shape))
print("Validation design matrix" + str(VAL_PHI.shape))
print("Testing design matrix" +str(TEST_PHI.shape))
print("Closed form weights are:" + str(W))
print("Calculating target values for all the datasets and also calculationg the ERMS values")
TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
# print(TR_TEST_OUT)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
# print(VAL_TEST_OUT)
TEST_OUT     = GetValTest(TEST_PHI,W)
# print(TEST_OUT)

TrainingAccuracy   = GetErms(TR_TEST_OUT,trainingTarget)
ValidationAccuracy = GetErms(VAL_TEST_OUT,validationTarget)
TestAccuracy       = GetErms(TEST_OUT,testingTarget)

print ('UBITname      = ysaraf')
print ('Person Number = 50290453')
print ('----------------------------------------------------')
print ("-----------Handwriting features dataset-------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("Number of Basis Function: " +str(M))
print ("Regularizer Constant (Lambda) :" + str(C_Lambda))
print ("E_rms Training   = " + str(TrainingAccuracy[1]))
print ("Training Accuracy = " + str(TrainingAccuracy[0]))
print ("E_rms Validation = " + str(ValidationAccuracy[1]))
print ("Validation Accuracy = " + str(ValidationAccuracy[0]))
print ("E_rms Testing    = " + str(TestAccuracy[1]))
print ("Testing Accuracy = " + str(TestAccuracy[0]))

# ## Gradient Descent solution for Linear Regression
print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')

## Random initialization of weights is done. The extra 1 term has been added for the bias.
W_Now        = np.random.rand(M+1,)
La           = C_Lambda
L_Erms_Val   = []
L_Accuracy_Val = []
L_Erms_TR    = []
L_Accuracy_TR    = []
L_Erms_Test  = []
L_Accuracy_Test  = []
W_Mat        = []
print("These iterations might take some significant time depending upon the number of basis function taken")
for i in range(len(TRAINING_PHI)):

    print ('---------Iteration: ' + str(i) + '--------------')
    prediction = np.dot(np.transpose(W_Now),TRAINING_PHI[i])
    Delta_E_D     = -np.dot((trainingTarget[i] - prediction),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next

    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next)
    Erms_TR       = GetErms(TR_TEST_OUT,trainingTarget)
    L_Erms_TR.append(float(Erms_TR[1]))
    L_Accuracy_TR.append(float(Erms_TR[0]))
    print("Training Target: " + str(trainingTarget[i]) + " ,Prediction:" + str(prediction))
    print("Train Accuracy :" + str(Erms_TR[0]) + " ,Train ERMS:" + str(Erms_TR[1]))

    # -----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next)
    Erms_Val      = GetErms(VAL_TEST_OUT,validationTarget)
    L_Erms_Val.append(float(Erms_Val[1]))
    L_Accuracy_Val.append(float(Erms_Val[0]))
    print("Val Accuracy :" + str(Erms_Val[0]) + " ,Val ERMS:" + str(Erms_Val[1]))

TEST_OUT_GD          = GetValTest(TEST_PHI,W_Now)
TestAccuracyGD       = GetErms(TEST_OUT_GD,testingTarget)
print ("E_rms Testing    = " + str(TestAccuracyGD[1]))
print ("Testing Accuracy = " + str(TestAccuracyGD[0]))

print ('----------Gradient Descent Solution--------------------')
print ("Number of Basis Function: " + str(M))
print ("Learning Rate used: " + str(learningRate))
print ("Regularization constant: " + str(La))
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))

## Plots for the training and validation sets are made for ERMS and accuracy.
plt.subplot(2, 2, 1)
plt.plot(L_Erms_TR)
plt.title('Training ERMS')
plt.xlabel('Number of datapoints fed:')

plt.subplot(2, 2, 2)
plt.plot(L_Erms_Val)
plt.title("Validation erms")
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

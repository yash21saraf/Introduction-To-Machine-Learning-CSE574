# coding: utf-8

from sklearn.cluster import KMeans
import numpy as np
import csv
import math
from matplotlib import pyplot as plt
# ## Defining the hyperparameters
C_Lambda = 2
learningRate = 0.01
# ## Defining each of the data splits
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
# ## Defining the number of clusters.
M = 10
PHI = []



# ## Here the t array has been created first and then as we are reading the rows i.e. the target value
# ## using for loop.
# ## Here we have a vector as each datapoint only corresponds to one target value.

def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return np.array(t)


# ## Here the dataMatrix array has been created first and then as we are reading the data rows
# ## using nested for loop.
# ## Here we first have all the data with the columns representing the feature space i.e. 46 features
# ## and the rows represent datapoints.
# ## Generating Datamatrix by reading all the features except the columns 5,6,7,8,9 as these
# ## has 0 variance. Also the matrix is not invertible when it has all a colums or row with zero
# ## variance.
# ## Then this matrix has been transposed. Now we have the feature space along the rows and datapoints along the columns.

def GenerateRawData(filePath):
    dataMatrix = []
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)

    dataMatrix = np.delete(dataMatrix, [5,6,7,8,9,45], axis=1)
    dataMatrix = np.transpose(dataMatrix)
    #print ("Data Matrix Generated..")
    return dataMatrix

# ## For training data the first TrainingPercent percentage of the Raw data is being assigned to the
# ## Training data.

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

# ## For training Target the first TrainingPercent percentage of the Raw data is being assigned to the
# ## Training Target.

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

# ## The TrainingCount is maintaining the last datapoint count which was associated with the
# ## training data.
# ## Here the data after the training data is being assigned to the validation split.
# ## The same function has been utilized to create the testing data split as the training count value is updated
# ## and then passed.
def GenerateValData(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")
    return dataMatrix

# ## The TrainingCount is maintaining the last datapoint count which was associated with the
# ## training data.
# ## Here the data after the training target is being assigned to the validation split.
# ## The same function has been utilized to create the testing data split as the training count value is updated
# ## and then passed.

def GenerateValTargetVector(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

# ## Here the trainingLen corresponds to the number of datapoints used and the len(DataT)
# ## gives the number of features.

# ## So first a vector vec is created for each feature.
# ## Each of this vector is then passed to the np.var which calculates the variance for
# ## each feature.
# ## This variance is attached to the new vector named varVect.
# ## Now this varVect contains all the variance terms.
# ## Using this vector the Bigsigma matrix is created which is a square matrix
# ## At the diagonal of this matrix the self-variance of all the features.

# ## Since the values for variance are very small the BigSigma has been multiplied
# ## by 200 to scale up the values. Here all the sef variances are multiplied by 200
# ## so there is no issues.

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

    BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma


# ## This function is used to calculate the weights for the closed form solution .
# ## The regularization term lamda is used.
# ## The Lamda matrix is a aquare matrix of dimensions equal to the number of basis functions.
# ## The Moore Pensrose Inversion is used here and a regularization term is added.
# ## The weights calculated are then returned.
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
    ##print ("Training Weights Generated..")
    return W
# ## This function is used to calculate the Phi matrix for all the datasets.
# ## The data is first transposed.
# ## The length i.e. the number of datapoints are then stored in the variable TrainingLen
# ## The PHI matrix is defined of dimensions as the number of rows as the number of datapoints
# ## and the number of columns as the number of basis functions.
# ## Since the formula for calculation of the RBF we require the Inverse of the BigSigma Matrix we take the
# ## matrix inverse.
# ## For each basis function and training point the basis function is calculated using the function
# ## GetRadialBasisOut
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    bias = np.ones((int(len(PHI)), 1))
    PHI = np.append(PHI, bias, 1)
    return PHI

# ## This function simply applies the formula for the calculation of the RBF and returns the scalar value.
# ## Here we have all three values i.e. DataRow, MuRow and the BigSigInv as the matrices.
# ## Using the getScalar function the value for the RBF is obtained for a particular datapoint and RBF.
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

# ## Matrix calculations are performed and the scalar value is returned.
# ## The functions involved are explained in the Report.
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

# ## To have a measure of how well the model has performed the ERMS or the root mean square value and the
# ## accuracy is being calculated.
# ## This is done using the expected target values and the output obtained.
# ## The difference between the two is calculated and squared.
# ## The square root of all the error values squared is taken in the end.
# ## Also a counter is maintained to calculate the number of correct predictions and then the accuracy of the
# ## model is calculated.

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    c1 = 0.0
    c0 = 0.0
    c2 = 0.0
    c0p = 0.0
    c1p = 0.0
    c2p = 0.0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if ValDataAct[i] == 0:
            c0 += 1
        if ValDataAct[i] == 1:
            c1 += 1
        if ValDataAct[i] == 2:
            c2 +=1
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
            if ValDataAct[i] == 0:
                c0p += 1
            if ValDataAct[i] == 1:
                c1p += 1
            if ValDataAct[i] == 2:
                c2p += 1
    c0acc = (float((c0p * 100)) / float(c0))
    c1acc = (float((c1p * 100)) / float(c1))
    c2acc = (float((c2p * 100)) / float(c2))

    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return [str(accuracy),str(math.sqrt(sum/len(VAL_TEST_OUT))), str(c0acc), str(c1acc),  str(c2acc)]


# ## Fetch and Prepare Dataset
# ## Fetching features and passing to GenerateRawData
# ## Fetching expected output and passing to GetTargetVector

RawTarget = GetTargetVector('Querylevelnorm_t.csv')
RawData   = GenerateRawData('Querylevelnorm_X.csv')
print ('----------------------------------------------------')
print("Shape of Raw feature data" + str(RawData.shape))
print("Shape of Raw Target data" + str(RawTarget.shape))
print ('----------------------------------------------------')
# ## The data splits are directly assigned in a sequential manner to the trainibg, validation
# ## and the testing set as the data is already randomized and not in sequence.
# ## So here the data can be directly assigned.

# ## Prepare Training Data
# ## The total data now needs to be split into training data, testing data and validation data.
# ## So both the input feature set and the target vector has been passed to create the required split.

TrainingTarget = GenerateTrainingTarget(RawTarget,TrainingPercent)
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print("Shape of Training Target data" + str(TrainingTarget.shape))
print("Shape of Training feature data" + str(TrainingData.shape))
print ('----------------------------------------------------')

# ## Prepare Validation Data
# ## Passing the length of training data is necessary to ensure that the data in the validation
# ## split starts after the end of the testing data.

ValDataAct = GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget)))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print("Shape of Validation target data" + str(ValDataAct.shape))
print("Shape of Validation feature data" + str(ValData.shape))
print ('----------------------------------------------------')

# ## Prepare Test Data
# ## The data after the training data and validation set is the testing set.
TestDataAct = GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print("Shape of Test target data" + str(TestDataAct.shape))
print("Shape of Test feature data" + str(TestData.shape))
print ('----------------------------------------------------')

# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

ErmsArr = []
AccuracyArr = []

# ## From sklearn a clustering algorithm i.e. the Kmeans clustering is imported.
# ## To the Kmeans functions some parameters has been passed which are
# ## M ie.e the number of clusters
# ## random_state is initialized to 0 i.e. to ensure reproduciblity of the same results
# ## The training data split transpose i.e. the features along columns and the datapoints in rows.

# ## The kmeans algorithm takes the default value for all the other parameters i.e.
# ## KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
# ##    n_clusters=10, n_init=10, n_jobs=1, precompute_distances='auto',
# ##    random_state=0, tol=0.0001, verbose=0)

# ## The dimensionality of the problem is still 41 so the kmeans is used to findout the cluster centers.
# ## kmeans.cluster_centers_ returns 10 centers each of which has a dimensionality of (41,1)

kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_

# ## BigSigma is a square matrix which contains all the self-variance terms of the features as a diagonal matrix.

BigSigma     = GenerateBigSigma(TrainingData)

# ## Now since we have the mean and variances for all the basis function the Phi matrix has been calculated for
# ## training set, testing set and validation set.

TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100)
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)

# ## Using the Training phi, Target and regularization term the weights are predicted.
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda))

print("Shape of feature center matrix" + str(Mu.shape))
print("Shape of Variance matrix" + str(BigSigma.shape))
print("Training design matrix" + str(TRAINING_PHI.shape))
print("Shape of weights calculated" + str(W.shape))
print("Validation design matrix" + str(VAL_PHI.shape))
print("Testing design matrix" +str(TEST_PHI.shape))

# ## Using the weights obtained using the Moore Penrose equation.
# ## These weights are then used to calculate the accuracy on all three data sets.
# ## So here first the outputs are calculated for each sets.
TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

# ## Finding Erms on training, validation and test set
# ## After calculating weights, used the same weights to find out the root mean square error
# ## on the testing and validation sets.

TrainingAccuracy   = GetErms(TR_TEST_OUT,TrainingTarget)
ValidationAccuracy = GetErms(VAL_TEST_OUT,ValDataAct)
TestAccuracy       = GetErms(TEST_OUT,TestDataAct)

print ('UBITname      = ysaraf')
print ('Person Number = 50290453')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("Number of Basis Function: " +str(M))
print ("Regularizer Constant (Lambda) :" + str(C_Lambda))
print ("E_rms Training   = " + str(TrainingAccuracy[1]))
print ("Training Accuracy = " + str(TrainingAccuracy[0]))
print ("0 acc: " + str(TrainingAccuracy[2]) + ", 1 acc: " + str(TrainingAccuracy[3]) + ", 2 acc: " + str(TrainingAccuracy[4]))
print ("E_rms Validation = " + str(ValidationAccuracy[1]))
print ("Validation Accuracy = " + str(ValidationAccuracy[0]))
print ("0 acc: " + str(ValidationAccuracy[2]) + ", 1 acc: " + str(ValidationAccuracy[3]) + ", 2 acc: " + str(ValidationAccuracy[4]))
print ("E_rms Testing    = " + str(TestAccuracy[1]))
print ("Testing Accuracy = " + str(TestAccuracy[0]))
print ("0 acc: " + str(TestAccuracy[2]) + ", 1 acc: " + str(TestAccuracy[3]) + ", 2 acc: " + str(TestAccuracy[4]))

# ## Gradient Descent solution for Linear Regression
print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')

W_Now        = np.dot(220, W)
# ## Random initialization of weights is done. The extra 1 term has been added for the bias.
#W_Now        = np.random.rand(M+1,)
La           = C_Lambda
L_Erms_Val   = []
L_Accuracy_Val = []
L_Erms_TR    = []
L_Accuracy_TR    = []
L_Erms_Test  = []
L_Accuracy_Test  = []
W_Mat        = []


# ## Here the data points are being processed one at a time
# ## After running each datapoint the weights are being updated.
# ## The regularization term of La is also added to avoid any overfitting
# ## The training and validation accuracy is calculated after each datapoint
for i in range(0,100):

    print ('---------Iteration: ' + str(i) + '--------------')
    # ## The output is predicted based on the previous weights
    prediction = np.dot(np.transpose(W_Now),TRAINING_PHI[i])
    # ## Based on the prediction the gradient is calculated.
    # ## Gradient also contains the regularization term
    Delta_E_D     = -np.dot((TrainingTarget[i] - prediction),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)
    # ## The net gradient is multiplied with the learning rate, i.e. scaling of gradient
    Delta_W       = -np.dot(learningRate,Delta_E)
    # ## The weights are updated based on the gradient
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next


    # ## The validation and training ERMS Are calculated based on the weights
    # ## Both the values are appended to the array so that it can be plotted.
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next)
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR[1]))
    L_Accuracy_TR.append(float(Erms_TR[0]))
    print("Training Target: " + str(TrainingTarget[i]) + " ,Prediction:" + str(prediction))
    print("Train Accuracy :" + str(Erms_TR[0]) + " ,Train ERMS:" + str(Erms_TR[1]))
    print("0 acc: " + str(Erms_TR[2]) + ", 1 Acc: " +str(Erms_TR[3] + ", 2 Acc: " + str(Erms_TR[4])))

    # -----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next)
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val[1]))
    L_Accuracy_Val.append(float(Erms_Val[0]))
    print("Val Accuracy :" + str(Erms_Val[0]) + " ,Val ERMS:" + str(Erms_Val[1]))

TEST_OUT_GD          = GetValTest(TEST_PHI,W_Now)
TestAccuracyGD       = GetErms(TEST_OUT_GD,TestDataAct)
print ("E_rms Testing    = " + str(TestAccuracyGD[1]))
print ("Testing Accuracy = " + str(TestAccuracyGD[0]))

print ('----------Gradient Descent Solution--------------------')
print ("Number of Basis Function: " + str(M))
print ("Learning Rate used: " + str(learningRate))
print ("Regularization constant: " + str(La))
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))

# ## Plots for the training and validation sets are made for ERMS and accuracy.
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
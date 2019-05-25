import pandas as pd
from keras.utils import np_utils


# ## Logic Based FizzBuzz Function [Software 1.0]

def fizzbuzz(n):
    
    # Logic Explanation
    if n % 3 == 0 and n % 5 == 0:
        return 'fizzbuzz'
    elif n % 3 == 0:
        return 'fizz'
    elif n % 5 == 0:
        return 'buzz'
    else:
        return 'other'


# ## Create Training and Testing Datasets in CSV Format

def createInputCSV(start,end,filename):
    
    # Why list in Python?
    
    # Answer - Lists is esentially a python array which is ordered and changable.
    # It has been used here as the list needs to be updated by using the .append method.
    inputData   = []
    outputData  = []
    
    # Why do we need training Data?
    
    # Answer - Here we are working on a classification problem. And for the neural network to
    # mathematically model the relation between the classes and the parameters we
    # need to pass multiple datapoints through it so that a point outside of these
    # datapoints is shown it maps it based on the generalized mathematical model.
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))
    
    # Why Dataframe?
    # Dataframes is included from pandas library. It allows indexing of the documents and also
    # add a label and can be easily pushed to the csv as and when required. 
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")


def processData(dataset):
    
    # Why do we have to process?
    
    # The processing is converting the discrete value to a 10 digit binary number.
    # Now advantage of having a binary input is either the neuron fires or it does'nt.
    # Having 10 binary features for a classification problem will make it easier for the network to map.
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel


# In[16]:


def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:

        # Why do we have number 10?
        
        # We have the dataset from 1-1000.
        # Since we are encoding the same to binary.
        # As 2^10 = 1024, we need at least 10 bits to represesnt each and every number in dataset.
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)



def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "fizzbuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel),4)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from matplotlib import pyplot

import numpy as np

input_size = 10
drop_out = 0.1
first_dense_layer_nodes  = 1024
second_dense_layer_nodes = 4

def get_model():
    
    # Why do we need a model?
    # Model is a represesntation of the generalized mathematical model of the given datapoints.
    # We need a model so that after training we could have relation between input and output
    # based on the model.
    
    # Why use Dense layer and then activation?
    # Dense layer simply means each neuron in the last layer is connected to the next layer
    # Dense implements the operation: output = activation(dot(input, kernel) + bias)
    # Dense layer is used for fully connected neural network.
    
    # Why use sequential model with layers?
    # Since we are using the simple architecture i.e. a fully conncected neural network it is
    # more suitable for our application.
    # Keras has two ways to build models i.e. sequential and functional.
    # But the functional is used to create more complex neural networks where any layer can be
    # connected to any other one.
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    # Why dropout?
    
    # When we pass the training data to the model, after multiple epochs the model starts to exactly repicate the
    # traing data rather than creating the generalized model.
    # Having the exact model might increses training accuracy but reduces the testing accuracy which is not what we desire.
    # So to avoid the above situation also known as overfitting we use dropouts i.e. dropping the neurons from some
    # layers in random manner. This helps us in avoiding overfitting.
    # Another solution to overfitting is using regularization functions which are used during the loss function calculation.
    # Usually a function of weights are added along with the loss function for regularization.
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    
    # Why Softmax?
    
    # Softmax is a layer used just before the output layer.
    # Based on the previous layer outputs the softmax layer assigns probablity to each class.
    # The size of softmax layer is same as the size of output layer.
    
    model.summary()
    
    # Why use categorical_crossentropy?
    
    # Categorial cross entropy isused for the evaluation of the loss function. For all the classification problems
    # the loss function used is cross entropy, and since we have more than 2 classes we have used categorial cross entropy.

    # Optimizer is the method used to find weight adjustments by working on the loss function.
    # Various optimizers available are SGD, Adagrad, adam, rmspop, adadelta
    
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')

model = get_model()


# The amount of data that is going to be used for validation among the training datatset.
validation_data_split = 0.2
num_epochs = 10000
model_batch_size = 64
tb_batch_size = 32
early_patience = 100


# Tensorboard has been used to visualize the model along with the graphs for multiple parameters to study the problem better.
tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)

# Early stopping is stopping the training sfter the loss starts to increase, this is done using a validation set
# The validation error starts to increase that means that the model is now generalized.
# Monitor defines the field that needs to be checked.
# patience argument represents the number of epochs before stopping once your loss starts to stops improving.
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')


# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )

# Used matplotlib pyplot to display all graphs.
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(5,7))
pyplot.show()


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "other"
    elif encodedLabel == 1:
        return "fizz"
    elif encodedLabel == 2:
        return "buzz"
    elif encodedLabel == 3:
        return "fizzbuzz"

wrong   = 0
right   = 0

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))
    
    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))

# Please input your UBID and personNumber 
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "ysaraf@buffalo.edu")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50290453")


predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel
output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')

print("UBID: ysaraf@buffalo.edu")
print("Person Nmumber: 50290453")

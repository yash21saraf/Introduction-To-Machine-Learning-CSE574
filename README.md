## Introduction to Machine Learning - CSE574

This repository contains all the projects performed as a part of coursework for CSE574. 

Course Page - [Introduction to Machine Learning: Course Materials](https://cedar.buffalo.edu/~srihari/CSE574/index.html)

-----
### Project-1.1 -

-----
The project is to compare two problem solving approaches to software development:
the logic-based approach (Software 1.0) and the machine learning approach (Software
2.0). It is also designed to quickly gain familiarity with Python and machine learning
frameworks.

The fizz buzz is a simple problem where,
- If number is divisible by 3, then print ”fizz”
- If number is divisible by 5, then print ”buzz”
- If number is divisible both by 3 and 5 , then print ”fizzbuzz”
- If number is not divisible both by 3 and 5 , then print ”others”

The simple classification problem has been solved by defining a single layer neural network in Keras. 

-----
### Project-1.2 - 

-----

The goal of this project is to use machine learning to solve a problem that arises in Information Retrieval,
one known as the Learning to Rank (LeToR) problem. We formulate this as a problem of linear regression
where we map an input vector x to a real-valued scalar target y(x, w).
There are two tasks:
1. Train a linear regression model on LeToR dataset using a closed-form solution.
2. Train a linear regression model on the LeToR dataset using stochastic gradient descent (SGD).

The LeToR training data consists of pairs of input values x and target values t. The input values are
real-valued vectors (features derived from a query-document pair). The target values are scalars (relevance
labels) that take one of three values 0, 1, 2: the larger the relevance label, the better is the match between
query and document. Although the training target values are discrete we use linear regression to obtain real
values which is more useful for ranking (avoids collision into only three possible values).

***Since the feature space is large. We also used RBF's to train the model. THe closed form solution is solved using Moore Penrose Equation.***

-----
### Project-2 -

-----

The project requires you to apply machine learning to solve the handwriting comparison task in forensics. We formulate this as a problem of linear regression where we map a set of input features x to a real-valued scalar target y(x,w).
The task is to find similarity between the handwritten samples of the known and the questioned writer by using linear regression.

Each instance in the CEDAR “AND” training data consists of set of input features for each hand-written “AND” sample. The features are obtained from two different sources:
1. Human Observed features: Features entered by human document examiners manually
2. GSC features: Features extracted using Gradient Structural Concavity (GSC) algorithm.

The target values are scalars that can take two values {1:same writer, 0:different writers}. Although the training target values are discrete we use linear regression to obtain real values which is more useful for finding similarity (avoids collision into only two possible values).

So for both the datasets the model has been trained and some inferences has been made. Now the dataset can be divided in three ways.

- Seen writer - When the writer is presesnt in both the training and testing dataset.
- Unseen writer - The writer is only present in the training dataset. The testing dataset contains all different writers.
- Shuffled - Where there is some overlap in the writers presesnt in training and testing set.
- 
Now these datasets have been created and then the Logistic Regression, Neural Networks performace has been compared in the report.

-----
### Project-3 -

-----

This project is to implement machine learning methods for the task of classification. You will first implement an ensemble of four classifiers for a given task. Then the results of the individual classifiers are combined to make a final decision.

The classification task will be that of recognizing a 28×28 grayscale handwritten digit image and identify it as a digit among 0, 1, 2, ... , 9. You are required to train the following four classifiers using MNIST digit images.

1. Logistic regression, which you implement yourself using backpropagtion and tune hyperparameters.
2. A publicly available multilayer perceptron neural network, train it on the MNIST digit images and tune hyperparameters.
3. A publicly available Random Forest package, train it on the MNIST digit images and tune hyperparameters.
4. A publicly available SVM package, train it on the MNIST digit images and tune hyperparameters.
5. Design a CNN to solve the same, and tune the hyperparameters. 
6. Use ensemble on all of them and make inferences. 

-----
### Project-4 -

-----

### Part-1-

The environment is designed as a grid-world 5x5:

![image](https://github.com/yash21saraf/Introduction-To-Machine-Learning-CSE574/blob/master/images/1.png)

- States - 25 possible states (0, 0), (0, 1), (0, 2), ... ,(4, 3), (4, 4)
- Actions - left, right, up, down
- The goal (yellow square) and the agent (green square) are dynamically changing the initial position on every reset.

![image](https://github.com/yash21saraf/Introduction-To-Machine-Learning-CSE574/blob/master/images/2.png)


#### Goal-
Our main goal is to let our agent learn the shortest path to the goal. In the environment the agent controls a green square, and the goal is to navigate to the yellow square (reward +1), using the shortest path. At the start of each episode all squares are randomly placed within a 5x5 grid-world. The agent has 100 steps to achieve as large a reward as possible. They have the same position over each reset, thus the agent needs to learn a fixed optimal path.

***Video Link for results-***

[![Model Performance](https://img.youtube.com/vi/Ao7MbP_edlA/0.jpg)](https://youtu.be/Ao7MbP_edlA)

### Part-2-

Implement DQN using at least two environments from OpenAI’s Gym library. You can use
Stable Baselines implementation of DQN, which provides a detailed documentation.

***Video Link for results-***

[![Model Performance](https://img.youtube.com/vi/S6v8ZslGe5E/0.jpg)](https://youtu.be/S6v8ZslGe5E)

import sys
import time

import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
import keras.datasets.mnist as data
from mnist import loadData

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(x):
    return np.sinh(x) / np.cosh(x)


def tanh_derivative(x):
    return 1.0 - tanh(x) ** 2


def relu(z):
    return np.maximum(z, 0)


def relu_derivative(x):
    return (x > 0)


def softmax(values):
    e_x = np.exp(values.T - np.max(values, axis=-1))
    return (e_x / e_x.sum(axis=0)).T


class MLP:
    def __init__(self, *layers, activation, learning_rate=0.01, batch_size=4, init_method = 'random'):
        self.learning_rate = learning_rate
        self.layers = layers
        self.activation = activation
        self.no_layers = len(layers)
        self.batch_size = batch_size
        self.bias = []
        self.theta_weights = []
        self.best_accuracy = 0
        self.best_acc_epoch = 0
        self.initialization_method = init_method
        self.initialize_weights()


    def initialize_weights(self):
        if self.initialization_method == 'random':
            for i in range(len(self.layers) - 1):
                self.theta_weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01)
                self.bias.append(np.random.randn(self.layers[i + 1]) * 0.01)

        if self.initialization_method == 'he':
            for i in range(len(self.layers) - 1):
                self.theta_weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2/self.layers[i]))
                self.bias.append(np.random.randn(self.layers[i + 1]) * 0.01)

        if self.initialization_method == 'xavier':
            for i in range(len(self.layers) - 1):
                self.theta_weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2/(self.layers[i] + self.layers[i+1])))
                self.bias.append(np.random.randn(self.layers[i + 1]) * 0.01)


    def feed_forward(self, X):
        g = None
        if self.activation == 'sigmoid':
            g = lambda x: sigmoid(x)

        if self.activation == 'relu':
            g = lambda x: relu(x)

        if self.activation == 'tanh':
            g = lambda x: tanh(x)

        A = [X]

        for i in range(len(self.theta_weights)):
            excitation = A[-1] @ self.theta_weights[i] + self.bias[i]

            if i == len(self.theta_weights) - 1:
                layer_activations = softmax(excitation)
            else:
                layer_activations = g(excitation)

            A.append(layer_activations)
        return A


    def test_accuracy(self, testX, testY):
        prediction = np.argmax(self.feed_forward(testX)[-1], axis=1)
        return np.mean(prediction ==  np.argmax(testY, axis=1))


    def backpropagation(self, X, Y):
        g = None
        if self.activation == 'sigmoid':
            g = lambda x: sigmoid_derivative(x)
        if self.activation == 'relu':
            g = lambda x: relu_derivative(x)
        if self.activation == 'tanh':
            g = lambda x: tanh_derivative(x)

        gradients = np.empty_like(self.theta_weights)
        A = self.feed_forward(X)

        errors = Y - A[-1]
        tmp = errors
        gradients[-1] = A[-2].T.dot(tmp)

        self.bias[-1] += np.sum(errors, axis=0) / len(errors)
        deltas = errors
        for i in range(len(A) - 2, 0, -1):
            deltas = g(A[i]) * deltas.dot(self.theta_weights[i].T)
            gradients[i - 1] = A[i - 1].T.dot(deltas)
            self.bias[i - 1] = np.sum(deltas, axis=0) / deltas.shape[1]

        self.theta_weights += self.learning_rate * gradients / len(X)

    def train(self, trainX, trainY, valX, valY, epochs):

        train_accuracy = []
        val_accuracy_list = []

        for i in range(epochs):
            for j in range(0, len(trainX), self.batch_size):
                X, Y = trainX[j:j + self.batch_size], trainY[j:j + self.batch_size]
                self.backpropagation(X, Y)

            prediction_Train = np.argmax(self.feed_forward(trainX)[-1], axis=1)
            prediction_Val = np.argmax(self.feed_forward(valX)[-1], axis=1)

            val_accuracy = np.mean(prediction_Val == np.argmax(valY, axis=1))

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_acc_epoch = i
            train_accuracy.append(np.mean(prediction_Train == np.argmax(trainY, axis=1)))
            val_accuracy_list.append(val_accuracy)



        return train_accuracy, val_accuracy_list


iterations = 1
trainX, trainY, testX, testY, valX, valY = loadData()

for i in range(iterations):

    model = MLP(784,128,10, activation='relu', learning_rate=0.1, batch_size=24, init_method='he')
    start_time = time.time()
    train_acc = model.train(trainX, trainY, valX, valY, 40)
    time_elapsed = time.time() - start_time
    test_acc = model.test_accuracy(testX, testY)

    print('Time elapsed: ', time_elapsed)
    print('Train accuracy: ', train_acc[-1])
    print("Best val accuracy: ", model.best_accuracy)
    print("Best val accuracy epoch:", model.best_acc_epoch)
    print('Test accuracy: ', test_acc)


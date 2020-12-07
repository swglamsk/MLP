import sys
import time

import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
# import keras.datasets.mnist as data
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


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


class MLP:
    def __init__(self, *layers, activation, learning_rate=0.01, batch_size=4, init_method='random', opti_method='none',
                 momentum=0.9, beta1=0, beta2=0):
        self.learning_rate = learning_rate
        self.layers = layers
        self.activation = activation
        self.no_layers = len(layers)
        self.batch_size = batch_size
        self.bias = []
        self.theta_weights = []
        self.best_accuracy = 0
        self.best_acc_epoch = 0
        self.opti_method = opti_method
        self.initialization_method = init_method
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.initialize_weights()
        self.all_weights_sqrt = np.asarray(
            [np.zeros(self.theta_weights[i].shape) for i in range(len(self.theta_weights))])
        self.all_bias_sqrt = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])
        self.prev_weights_updates = np.asarray(
            [np.zeros(self.theta_weights[i].shape) for i in range(len(self.theta_weights))])
        self.prev_bias_updates = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])

        self.all_weight_gradient = np.asarray(
            [np.zeros(self.theta_weights[i].shape) for i in range(len(self.theta_weights))])
        self.all_bias_gradient = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])

        self.all_weight_gradient_sqr = np.asarray(
            [np.zeros(self.theta_weights[i].shape) for i in range(len(self.theta_weights))])
        self.all_bias_gradient_sqr = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])

        self.accumulated_weight_updates = np.asarray(
            [np.zeros(self.theta_weights[i].shape) for i in range(len(self.theta_weights))])
        self.accumulated_bias_updates = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])

        self.accumulated_weight_gradients_sqr = np.asarray(
            [np.zeros(self.theta_weights[i].shape) for i in range(len(self.theta_weights))])
        self.accumulated_bias_gradients_sqr = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])

    def initialize_weights(self):
        if self.initialization_method == 'random':
            for i in range(len(self.layers) - 1):
                self.theta_weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01)
                self.bias.append(np.random.randn(self.layers[i + 1]) * 0.01)

        if self.initialization_method == 'he':
            for i in range(len(self.layers) - 1):
                self.theta_weights.append(
                    np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i]))
                self.bias.append(np.random.randn(self.layers[i + 1]) * 0.01)

        if self.initialization_method == 'xavier':
            for i in range(len(self.layers) - 1):
                self.theta_weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(
                    2 / (self.layers[i] + self.layers[i + 1])))
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
        return np.mean(prediction == np.argmax(testY, axis=1))

    def backpropagation(self, X, Y):
        g = None
        if self.activation == 'sigmoid':
            g = lambda x: sigmoid_derivative(x)
        if self.activation == 'relu':
            g = lambda x: relu_derivative(x)
        if self.activation == 'tanh':
            g = lambda x: tanh_derivative(x)

        opti = None
        if self.opti_method == 'momentum':
            opti = lambda x, y: self.momentum_f(x, y)
        if self.opti_method == 'nestrov':
            opti = lambda x, y: self.nestrov(x, y)
        if self.opti_method == 'adagrad':
            opti = lambda x, y: self.adagrad(x, y)
        if self.opti_method == 'adadelta':
            opti = lambda x, y: self.adadelta(x, y)
        if self.opti_method == 'adam':
            opti = lambda x, y: self.adam(x, y)
        if self.opti_method == 'none':
            opti = lambda x, y: self.none(x, y)

        gradients = np.empty_like(self.theta_weights)
        bias = np.empty_like(self.bias)

        A = self.feed_forward(X)

        errors = Y - A[-1]
        tmp = errors
        gradients[-1] = A[-2].T.dot(tmp)
        bias[-1] = np.sum(errors, axis=0) / len(errors)
        deltas = errors
        for i in range(len(A) - 2, 0, -1):
            deltas = g(A[i]) * deltas.dot(self.theta_weights[i].T)
            gradients[i - 1] = A[i - 1].T.dot(deltas)
            bias[i - 1] = np.sum(deltas, axis=0) / deltas.shape[1]
        # update weights

        gradients /= len(X)
        opti(gradients, bias)
        # update weights

    def train(self, trainX, trainY, valX, valY, epochs):

        train_accuracy = []
        val_accuracy_list = []
        train_loss_function = []
        val_loss_function = []

        for i in range(epochs):
            for j in range(0, len(trainX), self.batch_size):
                X, Y = trainX[j:j + self.batch_size], trainY[j:j + self.batch_size]
                self.backpropagation(X, Y)
            tmp_train = self.feed_forward(trainX)[-1]
            tmp_val = self.feed_forward(valX)[-1]
            prediction_Train = np.argmax(tmp_train, axis=1)
            prediction_Val = np.argmax(tmp_val, axis=1)

            val_accuracy = np.mean(prediction_Val == np.argmax(valY, axis=1))
            train_loss = cross_entropy(tmp_train, trainY)

            val_loss = cross_entropy(tmp_val, valY)

            train_loss_function.append(train_loss)
            val_loss_function.append(val_loss)

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_acc_epoch = i
            train_accuracy.append(np.mean(prediction_Train == np.argmax(trainY, axis=1)))
            val_accuracy_list.append(val_accuracy)

        return train_accuracy, val_accuracy_list, train_loss_function, val_loss_function

    def none(self, gradients, bias):
        self.theta_weights += self.learning_rate * gradients
        self.bias += self.learning_rate * bias

    def adam(self, gradients, bias):
        eps = 1e-08
        step_count = 1

        self.all_weight_gradient = (self.beta1 * self.all_weight_gradient + (1 - self.beta1) * gradients)
        self.all_bias_gradient = (self.beta1 * self.all_bias_gradient + (1 - self.beta1) * bias)

        self.all_weight_gradient_sqr = self.beta2 * self.all_weight_gradient_sqr + (1 - self.beta2) * gradients ** 2
        self.all_bias_gradient_sqr = self.beta2 * self.all_bias_gradient_sqr + (1 - self.beta2) * bias ** 2

        corrected_weight_gradient = self.all_weight_gradient / (1 - self.beta1 ** step_count)
        corrected_bias_gradient = self.all_bias_gradient / (1 - self.beta1 ** step_count)

        corrected_weight_gradient_sqrt = self.all_weight_gradient_sqr / (1 - self.beta2 ** step_count)
        corrected_bias_gradient_sqrt = self.all_bias_gradient_sqr / (1 - self.beta2 ** step_count)

        sqrt_weight_gradient_sqrt = np.empty_like(corrected_weight_gradient_sqrt)
        sqrt_bias_gradient_sqrt = np.empty_like(corrected_bias_gradient_sqrt)

        for i in range(len(self.all_weight_gradient_sqr)):
            sqrt_weight_gradient_sqrt[i] = np.sqrt(corrected_weight_gradient_sqrt[i]) + eps
            sqrt_bias_gradient_sqrt[i] = np.sqrt(corrected_bias_gradient_sqrt[i]) + eps

        self.theta_weights += self.learning_rate / sqrt_weight_gradient_sqrt * corrected_weight_gradient
        self.bias += self.learning_rate / sqrt_bias_gradient_sqrt * corrected_bias_gradient

    def adadelta(self, gradients, bias):
        eps = 1e-08
        step_count = 1

        # numerator
        rms_weight_updates = np.empty_like(self.theta_weights)
        rms_bias_updates = np.empty_like(self.bias)

        for i in range(len(self.accumulated_weight_gradients_sqr)):
            rms_weight_updates[i] = np.sqrt(self.accumulated_weight_updates[i] / step_count + eps)
            rms_bias_updates[i] = np.sqrt(self.accumulated_bias_updates[i] / step_count + eps)

        # denominator
        self.accumulated_weight_gradients_sqr = self.beta1 * self.accumulated_weight_gradients_sqr + (
                    1 - self.beta1) * (
                                                        gradients ** 2)
        self.accumulated_bias_gradients_sqr = self.beta1 * self.accumulated_bias_gradients_sqr + (1 - self.beta1) * (
                bias ** 2)

        rms_weight_gradients = np.empty_like(self.accumulated_weight_gradients_sqr)
        rms_bias_gradients = np.empty_like(self.accumulated_bias_gradients_sqr)

        for i in range(len(self.accumulated_weight_gradients_sqr)):
            rms_weight_gradients[i] = np.sqrt(self.accumulated_weight_gradients_sqr[i] / step_count + eps)
            rms_bias_gradients[i] = np.sqrt(self.accumulated_bias_gradients_sqr[i] / step_count + eps)

        #  Update
        current_weight_update = rms_weight_updates / rms_weight_gradients * gradients
        current_bias_update = rms_bias_updates / rms_bias_gradients * bias

        self.theta_weights += current_weight_update
        self.bias += current_bias_update

        self.accumulated_weight_updates = self.beta1 * self.accumulated_weight_updates + (1 - self.beta1) * (
                current_weight_update ** 2)
        self.accumulated_bias_updates = self.beta1 * self.accumulated_bias_updates + (1 - self.beta1) * (
                    current_bias_update ** 2)

    def adagrad(self, gradients, bias):
        learning_rate = 0.01
        eps = 1e-08

        self.all_weights_sqrt += gradients ** 2
        self.all_bias_sqrt += bias ** 2

        sqrt_weights_gradients = np.empty_like(self.all_weights_sqrt)
        sqrt_bias_gradients = np.empty_like(self.all_bias_sqrt)
        for i in range(len(self.all_weights_sqrt)):
            sqrt_weights_gradients[i] = np.sqrt(self.all_weights_sqrt[i] + eps)
            sqrt_bias_gradients[i] = np.sqrt(self.all_bias_sqrt[i] + eps)

        self.theta_weights += learning_rate / sqrt_weights_gradients * gradients
        self.bias += learning_rate / sqrt_bias_gradients * bias

    def nestrov(self, gradients, bias):

        gradients *= self.learning_rate
        bias *= self.learning_rate

        gradients += self.prev_weights_updates * self.momentum
        bias += self.prev_bias_updates * self.momentum

        self.theta_weights += gradients + gradients * self.momentum
        self.bias += bias + bias * self.momentum

        self.prev_weights_updates = np.copy(gradients)
        self.prev_bias_updates = np.copy(bias)

    def momentum_f(self, gradients, bias):

        gradients *= self.learning_rate
        bias *= self.learning_rate

        gradients += self.prev_weights_updates * self.momentum
        bias += self.prev_bias_updates * self.momentum

        self.theta_weights += gradients
        self.bias += bias

        self.prev_weights_updates = np.copy(gradients)
        self.prev_bias_updates = np.copy(bias)


iterations = 1
trainX, trainY, testX, testY, valX, valY = loadData()

for i in range(iterations):
    model = MLP(784, 128, 10, activation='sigmoid', learning_rate=0.01, batch_size=24, init_method='he',
                opti_method='momentum', momentum=0.7, beta1=0.9, beta2=0.99)
    start_time = time.time()
    train_acc, val_acc, train_loss, val_loss = model.train(trainX, trainY, valX, valY, 30)
    time_elapsed = time.time() - start_time
    test_acc = model.test_accuracy(testX, testY)

    print('Time elapsed: ', time_elapsed)
    print('Train accuracy: ', train_acc[-1])
    print("Best val accuracy: ", model.best_accuracy)
    print("Best val accuracy epoch:", model.best_acc_epoch)
    print('Test accuracy: ', test_acc)

    x = np.array(train_acc)
    y = np.array(val_acc)
    t_loss = np.array(train_loss)
    v_loss = np.array(val_loss)

    plt.plot(train_acc, label='train accuracy')
    plt.plot(val_acc, label='validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title("Optimalization: " + str(model.opti_method).upper() + " \n Initialization Method: " + str(
        model.initialization_method).upper())
    plt.legend(loc="upper left")
    plt.xticks(range(0, len(train_acc)))
    plt.yticks(np.arange(round(min(x.min(), y.min()), 2), 1, 0.01))
    plt.show()

    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(
        "Crossed Entropy \n Optimalization: " + str(model.opti_method).upper() + " \n Initialization Method: " + str(
            model.initialization_method).upper())
    plt.legend(loc="upper left")
    plt.xticks(range(0, len(train_acc)))
    plt.show()

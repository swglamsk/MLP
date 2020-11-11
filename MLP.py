import numpy as np
import gzip
import pickle as pk

def load_data():
    mnist_file = 'mnist.pkl.gz'
    with gzip.open(mnist_file, 'rb') as f:
        train_set, valid_set, test_set = pk.load(f, encoding='latin1')

    X = valid_set[0]
    y = valid_set[1]

    n_examples = len(y)
    labels = np.unique(y)
    Y = np.zeros((n_examples, len(labels)))
    for ix_label in range(len(labels)):
        ix_tmp = np.where(y == labels[ix_label])[0]
        Y[ix_tmp, ix_label] = 1

    return X, Y, labels, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    return np.exp(z) / sum(np.exp(z))


def matrix_to_vector(matrix):
    vector = np.array([])

    for one_layer in matrix:
        vector = np.concatenate((vector, one_layer.flatten("F")))
    return vector


class MLP:
    def __init__(self, layers, activation):
        self.lamda = 0
        self.layers = layers
        self.activation = activation
        self.no_layers = len(layers)
        self.theta_weights = self.initialize_weights()

    def initialize_weights(self):
        self.theta_weights = []
        size_layers = self.layers.copy()
        size_layers.pop(0)
        for size_layer, size_next_layer in zip(self.layers, size_layers):
            theta_tmp = np.random.randn(size_next_layer, size_layer + 1)
            self.theta_weights.append(theta_tmp)

        return self.theta_weights

    def feed_forward(self, X):
        g = None
        output = None
        if self.activation == 'sigmoid':
            g = lambda x: sigmoid(x)  # funkcja aktywacji

        A = [None] * self.no_layers  # aktywacje
        Z = [None] * self.no_layers  # pobudzenia

        input_layer = X

        for ix_layer in range(self.no_layers - 1):
            no_examples = input_layer.shape[0]
            input_layer = np.concatenate((np.ones([no_examples, 1]), input_layer), axis=1)
            A[ix_layer] = input_layer
            Z[ix_layer + 1] = np.matmul(input_layer, self.theta_weights[ix_layer].transpose())

            output = g(Z[ix_layer + 1])
            input_layer = output

        A[self.no_layers - 1] = output

        return A, Z

    def train(self, X, Y, iterations):
        for iteration in range(iterations):
            self.gradients = self.backpropagation(X, Y)
            self.gradients_vector = matrix_to_vector(self.gradients)
            self.theta_vector = matrix_to_vector(self.theta_weights)
            self.theta_vector = self.theta_vector - self.gradients_vector
            self.theta_weights = self.vector_to_matrix(self.theta_vector)

    def backpropagation(self, X, Y):
        g = None
        if self.activation == 'sigmoid':
            g = lambda x: sigmoid_derivative(x)

        A, Z = self.feed_forward(X)
        no_examples = X.shape[0]
        deltas = [None] * self.no_layers
        deltas[-1] = A[-1] - Y

        for ix_layers in np.arange(self.no_layers - 1 - 1, 0, -1):
            theta_tmp = self.theta_weights[ix_layers]

            theta_tmp = np.delete(theta_tmp, np.s_[0], 1)
            deltas[ix_layers] = (np.matmul(theta_tmp.transpose(), deltas[ix_layers + 1].transpose())).transpose() * g(
                Z[ix_layers])

        gradients = [None] * (self.no_layers - 1)
        for ix_layers in range(self.no_layers - 1):
            gradients_tmp = np.matmul(deltas[ix_layers + 1].transpose(), A[ix_layers])
            gradients_tmp = gradients_tmp / no_examples

            gradients_tmp[:, 1:] = gradients_tmp[:, 1:] + (self.lamda / no_examples) * self.theta_weights[ix_layers][:,
                                                                                       1:]

            gradients[ix_layers] = gradients_tmp

        return gradients

    def vector_to_matrix(self, vector):
        size_next_layers = self.layers.copy()
        size_next_layers.pop(0)

        matrix_list = []
        for size_layer, size_next_layer in zip(self.layers, size_next_layers):
            n_weights = size_next_layer * (size_layer + 1)
            data_tmp = vector[0: n_weights]
            data_tmp = data_tmp.reshape(size_next_layer, (size_layer + 1), order='F')
            matrix_list.append(data_tmp)
            vector = np.delete(vector, np.s_[0:n_weights])

        return matrix_list

    def predict(self, X):
        A, Z = self.feed_forward(X)
        Y_hat = A[-1]

        return Y_hat


epochs = 30
loss = np.zeros([epochs, 1])

X, Y, labels, y = load_data()
network = MLP([784, 100, 10], 'sigmoid')

for ix in range(epochs):
    network.train(X, Y, 10)
    Y_hat = network.predict(X)
    loss[ix] = (0.5)*np.square(Y_hat - Y).mean()


Y_hat = network.predict(X)
y_tmp = np.argmax(Y_hat, axis=1)
y_hat = labels[y_tmp]

acc = np.mean(1 * (y_hat == y))
print('Training Accuracy: ' + str(acc*100))


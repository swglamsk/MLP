from MLP import MLP
from convolution import Convolution, Pooling, Model
from mnist import loadData

if __name__ == '__main__':
    trainX, trainY, testX, testY, valX, valY = loadData()
    model = Model()

    model.add_layer(Convolution(16, 1, 3, 1))
    model.add_layer(Pooling(2, 2))
    model.add_layer(MLP(784, 128, 10, activation='sigmoid'))
    model.feed_forward(trainX[0])

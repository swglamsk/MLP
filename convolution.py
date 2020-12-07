import numpy as np


class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def feed_forward(self, image):
        current = image
        for layer in self.layers:
            result = layer.feed_forward(current)
            current = result
        return current


class Convolution:

    def __init__(self, num_of_filters, stride, size, frame_width):
        self.num_of_filters = num_of_filters
        self.stride = stride
        self.size = size
        self.frame_width = frame_width
        self.filters = np.random.randn(self.num_of_filters, size, size) * 0.01

    def feed_forward(self, image):
        print(image.shape)


class Pooling:

    def __init__(self, stride, size):
        self.stride = stride
        self.size = size
        self.input = None

    def feed_forward(self, image):
        self.input = image
        print(image.shape)

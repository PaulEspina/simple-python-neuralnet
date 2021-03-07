import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient(x):
    return x * (1 - x)


class NeuralNet:
    def __init__(self, n_input, n_hidden, n_output, activation=sigmoid):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.activation = np.vectorize(activation)
        self.gradient = np.vectorize(gradient)
        self.weights_input_hidden = np.random.uniform(-1, 1, (n_hidden, n_input))
        self.weights_hidden_output = np.random.uniform(-1, 1, (n_output, n_hidden))
        self.bias_hidden = np.random.rand(n_hidden, 1)
        self.bias_output = np.random.rand(n_output, 1)

    def feed_forward(self, inputs):
        inputs = np.matrix(inputs).reshape(self.n_input, 1)
        hiddens = np.matmul(self.weights_input_hidden, inputs)
        hiddens = hiddens + self.bias_hidden
        hiddens = self.activation(hiddens)
        outputs = np.matmul(self.weights_hidden_output, hiddens)
        outputs = outputs + self.bias_output
        return self.activation(outputs).tolist()

    def train(self, inputs, targets, learning_rate):
        inputs = np.matrix(inputs).reshape(self.n_input, 1)
        targets = np.matrix(targets).reshape(self.n_output, 1)
        hiddens = np.matmul(self.weights_input_hidden, inputs)
        hiddens = hiddens + self.bias_hidden
        hiddens = self.activation(hiddens)
        outputs = np.matmul(self.weights_hidden_output, hiddens)
        outputs = outputs + self.bias_output
        outputs = self.activation(outputs)

        outputs_error = targets - outputs
        outputs_gradient = self.gradient(outputs)
        outputs_gradient = outputs_gradient.T * outputs_error
        outputs_gradient = outputs_gradient * learning_rate
        weight_hidden_output_d = np.matmul(outputs_gradient, hiddens.T)

        self.weights_hidden_output = self.weights_hidden_output + weight_hidden_output_d
        self.bias_output = self.bias_output + outputs_gradient

        hiddens_error = np.matmul(self.weights_hidden_output.T, outputs_error)
        hiddens_gradient = self.gradient(hiddens)
        hiddens_gradient = hiddens_gradient.T * hiddens_error
        hiddens_gradient = hiddens_gradient * learning_rate
        weight_input_hidden_d = hiddens_gradient * inputs.T

        self.weights_input_hidden = self.weights_input_hidden + weight_input_hidden_d
        self.bias_hidden = self.bias_hidden + hiddens_gradient

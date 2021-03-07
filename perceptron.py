class Perceptron:
    def __init__(self, n_weights, learning_rate, bias, activation):
        self.learning_rate = learning_rate
        self.bias = bias
        self.activation = activation
        self.weights = []
        for i in range(n_weights + 1):
            self.weights.append(0)

    def feed_forward(self, inputs):
        weighted_sum = 0
        for i in range(len(self.weights) - 1):
            weighted_sum += self.weights[i] * inputs[i]
        weighted_sum += self.weights[len(self.weights) - 1]
        return self.activation(weighted_sum)

    def train(self, inputs, target):
        guess = self.feed_forward(inputs)
        error = target - guess
        for i in range(len(self.weights) - 1):
            self.weights[i] += self.learning_rate * error * inputs[i]
        self.weights[len(self.weights) - 1] += self.learning_rate * error * self.bias

    def get_weights(self):
        return self.weights

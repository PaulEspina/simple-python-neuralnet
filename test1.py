import multiprocessing
import random
import time

from neural_network import NeuralNet

random.seed(time.time())


class Train:
    def __init__(self):
        self.input = [random.randint(0, 1), random.randint(0, 1)]
        if self.input[0] ^ self.input[1]:
            self.label = [1]
        else:
            self.label = [0]


def learn(net, queue):
    for i in range(50000):
        train = Train()
        net.train(train.input, train.label, 0.01)
    queue.put(net)
    # for i in range(5000):
    #     train = Train()
    #     net.train(train.input, train.label, 0.05)
    # queue.put(net)
    # for i in range(15000):
    #     train = Train()
    #     net.train(train.input, train.label, 0.01)
    # queue.put(net)
    queue.put(1)


def log(queue):
    while True:
        net = queue.get()
        if net == 1:
            return
        # output = net.feed_forward([1, 1])
        # print('\n{:.2f}%'.format(output[0][0] * 100), '{:.2f}%'.format(output[1][0] * 100))
        # output = net.feed_forward([1, 0])
        # print('{:.2f}%'.format(output[0][0] * 100), '{:.2f}%'.format(output[1][0] * 100))
        # output = net.feed_forward([0, 1])
        # print('{:.2f}%'.format(output[0][0] * 100), '{:.2f}%'.format(output[1][0] * 100))
        # output = net.feed_forward([0, 0])
        # print('{:.2f}%'.format(output[0][0] * 100), '{:.2f}%'.format(output[1][0] * 100))
        output = net.feed_forward([1, 1])
        print('\n{:.2f}%'.format(output[0][0] * 100))
        output = net.feed_forward([1, 0])
        print('{:.2f}%'.format(output[0][0] * 100))
        output = net.feed_forward([0, 1])
        print('{:.2f}%'.format(output[0][0] * 100))
        output = net.feed_forward([0, 0])
        print('{:.2f}%'.format(output[0][0] * 100))


if __name__ == '__main__':
    q = multiprocessing.Queue()
    nn = NeuralNet(2, 4, 1)
    q.put(nn)
    learning1 = multiprocessing.Process(target=learn, args=(nn, q))
    printing = multiprocessing.Process(target=log, args=(q,))
    learning1.start()
    printing.start()
    learning1.join()
    printing.join()


import numpy as np
import random

import gzip
import cPickle
import matplotlib.pyplot as plt

# start of given code

def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    data = cPickle.load(f)
    f.close()
    return data


def plot_digits(data, numcols, shape=(28,28)):
    numdigits = data.shape[0]
    numrows = int(numdigits/numcols)
    for i in range(numdigits):
        plt.subplot(numrows, numcols, i)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    plt.show()

# end of given code


def logreg_gradient(x, t, w, b):
    """
        QUESTION 1.1.2

    returns gradients wrt w_j for each j, and b, over the Log Likelihood

    computation will roughly contain
        log q -> Z log p -> delta^q
    """
    z = 0
    logqs = []

    # first calculate all q_j's (log q_j = w_j ^ T x + b_j)
    # also calculate Z (= sum over all q_j's)
    for j in range(len(b)):
        logq_j = w[j].T.dot(x) + b[j]
        z += np.exp(logq_j)
        logqs.append(logq_j)

    # now we can calculate the grads
    grad_w = []
    grad_b = []

    for j, logq_j in enumerate(logqs):
        t_is_j = 1 if j == t else 0
        # equation states q_j, so we take the exp of the log
        delta_q_j = t_is_j - np.exp(logq_j)/z
        # delta_b_j = delta^q_j
        grad_b.append( delta_q_j )

        grad_w.append([])
        for i in xrange(np.shape(w)[1]):
            grad_w[j].append( delta_q_j * x[i] )


    # make np,matrices
    grad_w = np.matrix(grad_w)
    grad_b = np.array(grad_b)

    # return em as matrices
    return grad_w, grad_b

def sgd_iter(x_train, t_train, w, b):
    """
        QUESTION 1.1.3

    performs one iteration of stochastic gradient descent (SGD)
    and returns new weights.

    Goes to the training set once in random order and calls logreg_gradient
    for each datapoint to get the gradient and updates the parameters using a
    small learning rate (e.g. 1*10^-4). Note: since we are maximizing the
    likelihood this is actually gradient ascent.
    """

    # set learning rate to 1E-4
    eta = 0.0001

    # shuffle the data to go through the data in a random order
    data = zip(x_train, t_train)
    random.shuffle(data)

    for x, t in data:
        # get the gradient for current datapoint
        delta_w, delta_b = logreg_gradient(x, t, w, b)
        w += eta * delta_w
        b += eta * delta_b

        print 'b', b
        #print 'w', w
        print 'sumsum w', sum(sum(w))
    return w, b

if __name__ == '__main__':
    # given tests
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

    # uncomment for plot of the digits
    #plot_digits(x_train[0:8], numcols=4)

    # initialize w and b at zeroes
    w = np.zeros((10, 784))
    b = np.zeros(10)

    # perform 1 sgd iteration
    w, b = sgd_iter(x_train, t_train, w, b)

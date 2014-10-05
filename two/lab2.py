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

    returns gradients wrt w_j for each j, and b, over the Log (partial) Likelihood
    """
    z = 0
    logqs = []

    # first calculate all q_j's (log q_j = w_j ^ T x + b_j)
    # also calculate Z (= sum over all q_j's)
    for j in range(len(b)):
        logq_j = w[j].T.dot(x) + b[j]

        # note: np.exp in the sum for Z!!
        z += np.exp(logq_j)
        logqs.append(logq_j)

    # now we can calculate the grads
    grad_w = []
    grad_b = []

    for j, logq_j in enumerate(logqs):
        t_is_j = 1 if j == t else 0
        # equation states q_j, so we take the exp of the log
        # TODO: should be checked: is this the sum exp trick applied? We do
        # np.exp(logq_j - np.log(z)) instead of: np.exp(logq_j)/z
        delta_q_j = t_is_j - np.exp(logq_j - np.log(z))
        # delta_b_j = delta^q_j
        grad_b.append(delta_q_j)

        # FIXME: should be able to do this with a matrix calculation!
        grad_w.append([])
        for i in xrange(np.shape(w)[1]):
            grad_w[j].append( delta_q_j * x[i] )

    # make np,matrices
    grad_w = np.matrix(grad_w)
    grad_b = np.array(grad_b)

    # return em as matrices
    return grad_w, grad_b

def sgd_iter(x_train, t_train, w, b, verbose=False):
    """
        QUESTION 1.1.3

    performs one iteration of stochastic gradient descent (SGD)
    and returns new weights.

    Goes to the training set once in random order and calls logreg_gradient
    for each datapoint to get the gradient and updates the parameters using a
    small learning rate (e.g. 1*10^-4). Note: since we are maximizing the
    likelihood this is actually gradient ascent.

    ASSUMPTION: I was actually rather confused if the x_train and t_train
    variables contained the whole dataset or just 1 point. I assumed they
    contained the whole dataset

    returns: w, b

    Run this to test:
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
    w = np.zeros((10, 784))
    b = np.zeros(10)
    # perform 1 sgd iteration
    w, b = sgd_iter(x_train[:200], t_train[:200], w, b)
    """

    # set learning rate to 1E-4
    eta = 0.0001

    # shuffle the data to go through the data in a random order
    data = zip(x_train, t_train)
    random.shuffle(data)

    for i, (x, t) in enumerate(data):

        if verbose and i%50 == 0:
            print '(sgd_iter) iteration %d', i

        # get the gradient for current datapoint
        delta_w, delta_b = logreg_gradient(x, t, w, b)
        w += eta * delta_w
        b += eta * delta_b
    return w, b

def train_and_plot(verbose=False):
    """
        QUESTION 1.2.1
    """
    # set to a negative value to train each iteration on complete set
    subset_training_size = 50

    # load the heck out of the given data!!
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

    # initialize w and b at zeroes
    w = np.zeros((10, 784))
    b = np.zeros(10)

    train_errors = []
    validate_errors = []

    # a `handful' of iterations is anywhere between 2 and 5 iterations?
    #handful_of_iterations = random.randint(2, 5)
    handful_of_iterations = 100
    for i in xrange(handful_of_iterations):

        if verbose and i%10 == 0:
            print '(train_and_plot) iteration %d' % i

        # if training on a subset:
        # shuffle and select subset of data for 1 iteration (because otherwise
        # it takes too long and overfits immediately)
        if subset_training_size > 0:
            random.shuffle(x_train)
            random.shuffle(t_train)
            w, b = sgd_iter(x_train[:subset_training_size],
                t_train[:subset_training_size], w, b, verbose=verbose)
        else:
            w, b = sgd_iter(x_train, t_train, w, b)

        # TODO: plot conditional log-probability instead of #correctly classiefied
        # calculate error on the training a validation sets
        training_error = get_error(w, b, x_train, t_train)
        validate_error = get_error(w, b, x_valid, t_valid)
        train_errors.append(training_error)
        validate_errors.append(validate_error)

    # TODO: make pretty plot
    # plot
    plt.plot(range(len(train_errors)), train_errors)
    plt.plot(range(len(validate_errors)), validate_errors)
    plt.show()

def get_error(w, b, x, t):
    """ help function for train_and_plot. Calculates the amount of correctly
    classified instances of t using w and b."""

    correct = 0
    for x_i, t_i in zip(x, t):
        # get q
        qs = w.dot(x_i) + b
        # do not need to normalize, just take the index of the largest value
        correct += 1 if get_index_of_largest(qs) == t_i else 0
    return correct

def get_index_of_largest(ar):
    """ somehow I have to do this manually? :("""
    index = 0
    m = ar[index]
    for i, a in enumerate(ar[1:]):
        index = i if a > ar[index] else index
    return index

def visualize_weights():
    """
        QUESTION 1.2.2
    """
    # load the heck out of the given data!!
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

    # initialize w and b at zeroes
    w = np.zeros((10, 784))
    b = np.zeros(10)

    # train on 20% of data (100% = 50000)
    training_length = 10000;

    # perform 1 sgd iteration
    w, b = sgd_iter(x_train[:5000], t_train[:5000], w, b)

    # visualize weights
    plot_digits(w, 5)

if __name__ == '__main__':
    # load the heck out of the given data!!
    train_and_plot(verbose=True)

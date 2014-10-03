import numpy as np
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
        QUESTION 1.1.1

    returns gradients wrt w_j for each j, and b, over the Log Likelihood

    computation will roughly contain
        log q -> Z log p -> delta^q
    """
    z = 0
    qs = []

    # first calculate all q_j's (log q_j = w_j ^ T x + b_j)
    # also calculate Z (= sum over all q_j's)
    for j in range(len(b)):
        q_j = w.get(j).dot(x) + b.get(j)
        z += q_j
        qs.append(q_j)

    # now we can calculate the grads
    grad_w = []
    grad_b = []

    for j, q_j in enumerate(qs):
        t_is_j = 1 if j == t else 0
        grad_b.append( t_is_j - q_j/z) # (1) - delta^q_j = q_j / sum q
        grad_w.append( t_is_j - grad_b[-1] * 1./ w.get(j).dot(x) + grad_b[-1])

    grad_w = grad_w.dot(x)

    return grad_w, grad_b

if __name__ == '__main__':
    # given tests
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
    plot_digits(x_train[0:8], numcols=4)

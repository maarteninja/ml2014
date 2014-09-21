from pylab import *
import math
import matplotlib.pyplot as plt
import numpy as np


def gen_sinusoidal(n):
    '''
        QUESTION 1.1

    Generates toy data like in MLPR book
    Returns N-dimensional vectors x and t, where 
    x contains evenly spaced values from 0 to 2pi
    and elements ti of t are distributed according
    to ti ~ N(mean, variance) where xi is the ith
    element of x, the mean = sin(xi) and
    standard deviation = 0.2

    x, t = gen_sinusoidal(10)
    '''
    x = linspace(0, 2*math.pi, n)
    t = []
    sigma = 0.2
    for i in x:
        mu = math.sin(i)
        s = np.random.normal(mu, sigma)
        t.append(s)
    return x, array(t)

def create_phi(x, m):
    """
    x, t = gen_sinusoidal(10)
    create_phi(x, 5)
    """
    if m < 0:
        raise ValueError('m can not be negative')

    # plus one for the non-optional first element of the bias vector ^0
    m += 1
    # create array of exponents [0, 1, ..., m-1]
    phi = array(range(m))
    # reserve space for NxM design matrix
    Phi = zeros((size(x), m))

    for n, x_elem in enumerate(x):
        # create array filled with m copies of the nth datapoint
        x_ar = array([x_elem] * m)
        # multiply with the bias vector
        Phi[n] = x_ar ** phi
    return matrix(Phi)

def fit_polynomial(x, t, m):
    '''
        QUESTION 1.2

    Finds maximum-likelihood solution of 
    unregularized M-th order fit_polynomial for dataset x using t as the target vector.
    Returns w -> maximum-likelihood parameter
    estimates

    x, t = gen_sinusoidal(10)
    w = fit_polynomial(x, t, 3)
    '''

    phi = create_phi(x, m)

    # solve for w
    return phi.T.dot(phi).I.dot(phi.T).dot(t)

def plot_sine(linspace, label=None):
    t = array([math.sin(i) for i in linspace])
    if label:
        print label
        plt.plot(linspace, t, label=label)
    else:
        plt.plot(linspace, t)

def plot_polynomial(linspace, w, color='g', label=None):
    """ plots a function for a w over a given range for x"""

    # for each x: sum for all m:  w[m]*x**m
    f = [sum(w.item(p) * (x_point ** p) for p in range(size(w, 1))) for x_point in linspace]

    # make pretty plot
    if label:
        plt.plot(linspace, f, color=color, label=label)
    else:
        plt.plot(linspace, f, color=color)

def fit_polynomial_reg(x, t, m, lamb):
    """
    x, t = gen_sinusoidal(10)
    w = fit_polynomial_reg(x, t, 3)
    """
    Phi = create_phi(x, m)

    i = np.eye(np.size(Phi, 1))

    w = (lamb * i)
    w = (w + Phi.T.dot(Phi)).I
    w = w.dot(Phi.T).dot(t)
    return w

def kfold_indices(N, k):
    """ Given function to generate indices of cross-validation folds """
    all_indices = np.arange(N,dtype=int)
    np.random.shuffle(all_indices)
    idx = np.floor(np.linspace(0,N,k+1))
    train_folds = []
    valid_folds = []
    for fold in range(k):
        valid_indices = all_indices[idx[fold]:idx[fold+1]]
        valid_folds.append(valid_indices)
        train_folds.append(np.setdiff1d(all_indices, valid_indices))
    return train_folds, valid_folds


def model_selection_by_cross_validation(data=None, plot=True):
    """
        QUESTION 1.5

    Selects the optimal model using cross validation.

    The keyword argument data indicated whether data is given or should be
    generated (default=None, means it will be generated)

    The keyword argument plot indicates whether the errors for the different m
        and k is plotted (default=True).
    """
    n = k = 9
    if not(data):
        x, t = gen_sinusoidal(n)
    else:
        x, t = data

    indices = kfold_indices(n, k)

    min_error = np.inf
    max_error = -np.inf

    if plot:
        afig, ax = plt.subplots()

    # loop over m and lambda
    for lambda_exp in range(-10, 1):
        errors = []
        lamb = np.e ** lambda_exp
        for m in range(9):

            # set avg. error to 0 and calculate actual lambda value
            error = 0

            # loop over the folds
            for train, heldout in zip(*indices):

                # get the indices of the current fold
                xs = [x[i] for i in train]
                ts = [t[i] for i in train]

                # fit model, on selected points
                w = fit_polynomial_reg(xs, ts, m, lamb)
                #w = fit_polynomial(xs, ts, m)

                # get the value were going to predict on
                t_value = t[heldout[0]]
                x_value = x[heldout[0]]

                # predict: t = w0 * x ** 0 + w1 ** x ** 1 + ...
                prediction = [np.sum(w.item(p) * (x_value** p) for p in range(size(w, 1)))][0]

                error += .5 * float(prediction - t_value) ** 2 + (lamb/2.) * float(w.dot(w.T))

            errors.append(error)

            if error < min_error:
                min_error = error
                best_model = (m, lambda_exp)
            if error > max_error:
                max_error = error

        if plot:
            ax.plot(range(9), errors, label="lambda = e^" + str(lambda_exp))

    if plot:
        legend = ax.legend(loc='upper left')
        ax.set_ylim((0, max_error))
        ax.set_xlabel("m")
        ax.set_ylabel("average absolute error")
        ax.set_title("Error for lambda and m")

        plt.show()

    return best_model

def plot_best_cross_validated_fit():
    """
        QUESTION 1.6
    """
    n = 9
    x, t = gen_sinusoidal(n)
    best_m, best_lamb = model_selection_by_cross_validation(data=(x,t), plot=False)
    w = fit_polynomial_reg(x, t, best_m, best_lamb)

    print 'best_m', best_m, 'best lamb', best_lamb

    linspace = np.linspace(0, 2*math.pi, 1000)
    plot_polynomial(linspace, w, label='best fit')
    plot_sine(linspace, label='true sin')
    plot(x, t, 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('m = %d, lambda = e^%d' % (best_m, best_lamb))
    legend()
    plt.show()



def one_point_three():
    """
        QUESTION 1.3
    """

    N = 10
    orders = [0, 1, 3, 9]
    res = 1000

    x, t = gen_sinusoidal(N)
    linspace = np.linspace(0, 2*math.pi, res)

    for i, m in enumerate(orders):
        w = fit_polynomial(x, t, m)

        plt.subplot(2, 2, i+1)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.xlim(0, 2*math.pi)
        plt.ylim(-1.5, 1.5)
        plt.title('m=%d'%m)
        plt.tight_layout()

        plt.plot(x, t, 'o')

        plot_sine(linspace)
        plot_polynomial(linspace, w)

    plt.show()

if __name__ == '__main__':
    one_point_three()
    #print model_selection_by_cross_validation()
    #plot_best_cross_validated_fit()

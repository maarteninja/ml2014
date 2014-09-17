from pylab import *
import math
import matplotlib.pyplot as plt


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
        s = random.normal(mu, sigma)
        t.append(s)
    return x, array(t)

def create_phi(x, t, m):
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

    Phi = create_phi(x, t, m)

    # solve for w
    return Phi.T.dot(Phi).I.dot(Phi.T).dot(t)

def plot_sine(linspace):
    t = array([math.sin(i) for i in linspace])
    plt.plot(linspace, t)

def plot_polynomial(linspace, w):
    """ 
    generates 9 sinusoidal points with some noise. Fits 4 models polynomial
    using least squares solution for m = 0, 1, 3, 9. Pretty plots them all.
    """
    # for each x: sum for all m:  w[m]*x**m
    f = [sum(w.item(p) * (x_point ** p) for p in range(size(w, 1))) for x_point in linspace]

    # make pretty plot
    plt.plot(linspace, f)

def fit_polynomial_reg(x, t, m, lamb):
    """
    x, t = gen_sinusoidal(10)
    w = fit_polynomial(x, t, 3)
    """
    Phi = create_phi(x, t, m)

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

def model_selection_by_cross_validation():
    n = k = 9
    x, t = gen_sinusoidal(n)
    indices = kfold_indices(n, k)

    min_error = np.inf

    # loop over m and lambda
    for m in range(11):
        for lambda_exp in range(10, -1, -1):

            # set avg. error to 0 and calculate actual lambda value
            avg_error = 0
            lamb = np.e ** -lambda_exp

            # loop over the folds
            for train, heldout in zip(*indices):
                # get the indices of the current fold
                xs = [x[i] for i in train]
                ts = [t[i] for i in train]

                # fit model
                w = fit_polynomial_reg(xs, ts, m, lamb)

                # calculate error
                t_value = t[heldout[0]]
                x_value = x[heldout[0]]

                prediction = [np.sum(w.item(p) * (x_value** p) for p in range(m+1))]
                avg_error += (prediction - t_value)/k

        if avg_error < min_error:
            best_model = (m, lamb)
    return m, lamb

def run():

    N = 10
    orders = [0, 1, 3, 9]
    res = 1000

    x, t = gen_sinusoidal(N)
    linspace = linspace(0, 2*math.pi, res)

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
    run()

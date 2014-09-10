from lab1 import *

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

if __name__ == '__main__':
    #one_point_three_plot()
    #x, t = gen_sinusoidal(10)
    #w = fit_polynomial(x, t, 3)
    #print w
    print model_selection_by_cross_validation()


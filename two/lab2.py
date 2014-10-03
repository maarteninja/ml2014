import numpy as np

# COMPELETELY UNTESTE

def logreg_gradient(x, t, w, b):
    """ returns gradients wrt w_j for each j, and b, over the Log Likelihood

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

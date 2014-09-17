from pylab import *
import numpy as np
from lab1 import create_phi, plot_sine, plot_polynomial
import math
import matplotlib.pyplot as plt


def gen_sinusoidal2(n):
    x = 2*math.pi*(rand(1,n))[0]
    t = []
    sigma = 0.2
    for i in x:
        mu = math.sin(i)
        s = np.random.normal(mu, sigma)
        t.append(s)
    return x, array(t)

def fit_polynomial_bayes(x, t, M, alpha, beta):
    Phi = create_phi(x, t, M)
    N = size(Phi, 1)
    I = eye(N, N)

    Sn = (beta * Phi.T.dot(Phi) + alpha * I).I
    mn = beta * Sn.dot(Phi.T).dot(t)
    
    return Sn, mn

def question_2_4():
    N = 7
    M = 5
    alpha = 0.5
    beta = 1/0.2**2
    res = 1000

    x, t = gen_sinusoidal2(N)
    Sn, mn = fit_polynomial_bayes(x, t, M, alpha, beta)
    ls = linspace(0, 2*math.pi, res)
    plot_sine(ls)
    plot_polynomial(ls, mn)
    plt.plot(x, t, 'o')
    plt.show()

def predict_polynomial_bayes(x, m, S, beta)
	
    s = np.random.normal(mu, sigma)
    #p_mean = mTN ϕ(x) = 
    #p_var = 
    phi = []
    p_mean = []
    
    for i in len(m):
		phi.append(x^i)
    
    return p_mean, p_var

if __name__ == "__main__":
    question_2_4()


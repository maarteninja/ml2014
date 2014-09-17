import numpy as np
import math
import matplotlib.pyplot as plt


def gen_sinusoidal2(n):
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
    #x = np.linspace(0, 2*math.pi, n)
    x = 2*math.pi*(np.rand(1,n))
    sigma = 0.2
    for i in x:
        mu = math.sin(i)
        s = np.random.normal(mu, sigma)
        t.append(s)
    return x, np.array(t)


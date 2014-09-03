import numpy as np
import math

'''
Generates toy data like in MLPR book
Returns N-dimensional vectors x and t, where 
x contains evenly spaced values from 0 to 2pi
and elements ti of t are distributed according
to ti ~ N(mean, variance) where xi is the ith
element of x, the mean = sin(xi) and
standard deviation = 0.2
'''
def gen_sinusoidal(N):
	x = np.linspace(0, 2*math.pi, N)
	t = []
	sigma = 0.2
	for i in x:
		mu = math.sin(i)
		s = np.random.normal(mu, sigma)
		t.append(s)

gen_sinusoidal(10)
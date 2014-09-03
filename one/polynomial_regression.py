import numpy as np
import math

def gen_sinusoidal(N):
	'''
	Generates toy data like in MLPR book
	Returns N-dimensional vectors x and t, where 
	x contains evenly spaced values from 0 to 2pi
	and elements ti of t are distributed according
	to ti ~ N(mean, variance) where xi is the ith
	element of x, the mean = sin(xi) and
	standard deviation = 0.2
	'''
	x = np.linspace(0, 2*math.pi, N)
	t = []
	sigma = 0.2
	for i in x:
		mu = math.sin(i)
		s = np.random.normal(mu, sigma)
		t.append(s)
	t = np.array(t)
	return x, t

def fit_polynomial(x, t, M):
	'''
	Finds maximum-likelihood solution of 
	unregularized M-th order fit_polynomial
	for dataset x using t as the target vector.
	Returns w -> maximum-likelihood parameter
	estimates
	'''
	bias = np.linspace(0, M-1, M).reshape(1,M)
	x = x.reshape(1, len(x))
	design_matrix = np.power(x.T, bias)

def error_function(w, matrix):
	pass

vectors = gen_sinusoidal(10)
fit_polynomial(vectors[0], vectors[1], 5)
import numpy as np
import math
import matplotlib.pyplot as plt

def gen_sinusoidal(n):
	'''
	Generates toy data like in MLPR book
	Returns N-dimensional vectors x and t, where 
	x contains evenly spaced values from 0 to 2pi
	and elements ti of t are distributed according
	to ti ~ N(mean, variance) where xi is the ith
	element of x, the mean = sin(xi) and
	standard deviation = 0.2

	x, t = gen_sinusoidal(10)
	'''
	x = np.linspace(0, 2*math.pi, n)
	t = []
	sigma = 0.2
	for i in x:
		mu = math.sin(i)
		s = np.random.normal(mu, sigma)
		t.append(s)
	return x, np.array(t)

def fit_polynomial(x, t, m):
	'''
	Finds maximum-likelihood solution of 
	unregularized M-th order fit_polynomial
	for dataset x using t as the target vector.
	Returns w -> maximum-likelihood parameter
	estimates

	w = fit_polynomial(x, t, 3)
	'''
	phi = np.array(range(m))
	Phi = np.zeros((np.size(x), m))
	for i, x_elem in enumerate(x):
		x_ar = np.array([x_elem] * m)
		Phi[i] = x_ar ** phi
	Phi = np.matrix(Phi)
	return Phi.T.dot(Phi).I.dot(Phi.T).dot(t)

def one_point_three_plot():
	n = 9
	x, y = gen_sinusoidal(n)

	x_points = np.linspace(0, 2*math.pi, 1000)
	t = np.array([math.sin(i) for i in x_points])

	fig, ax = plt.subplots()

	for f, m in enumerate([1, 1, 3, 9]):
		print m
		w = np.array(fit_polynomial(x, y, m))
		#for i in range(np.size(w, 1)):
		#	w.itemset(i, w.item(i) ** i)
		g = np.array([np.sum(w * x_point ** np.array(range(m))) for x_point in x_points])

		ax = fig.add_subplot(2, 2, f, label='m=%d'%m)
		ax.plot(x_points, t)
		ax.plot(x, y, 'o')
		ax.plot(x_points, g)

	plt.show()

one_point_three_plot()


#def one_point_three():\n",                                                
#    \"\"\"Plot for 1.3\"\"\"\n",                                          
#    x, t = gen_sinusoidal(10)\n",                                         
#    fig, ax = plt.subplots()\n",                                          
#    plt.plot(x, t, '.')\n",                                               
#    plt.ylim((-2, 2))\n",                                                 
#    plt.xlim((0, 2*np.pi))\n",                                            
#    for m in [0, 1, 3, 9]:\n",                                            
#        w = fit_polynomial(x, t, m)\n",                                   
#        xs = np.arange(0, 2*np.pi, 0.01);\n",                             
#        ys = []\n",                                                       
#        for xi in xs:\n",                                                 
#            y = 0.0\n",                                                   
#            for i in range(m+1):\n",                                      
#                y += w.item(i) * (xi ** i)\n",                            
#            ys.append(y)\n",                                              
#        plt.plot(xs, ys, label=\"m = \" + str(m))\n",                     
#    xs = np.arange(0, 2*np.pi, 0.01)\n",                                  
#    ys = np.sin(xs)\n",                                                   
#    plt.plot(xs, ys, 'k', label=\"sinusoidal mean\")\n",                  
#    ax.set_title(\"Fittining Polynomials\")\n",                           
#    legend = ax.legend(loc='lower left', shadow=True)\n",                 
#    plt.show()\n",                                      

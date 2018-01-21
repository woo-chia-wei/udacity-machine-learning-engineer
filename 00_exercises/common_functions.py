import numpy as np
import matplotlib.pyplot as plt

def plot_model_with_boundaries(model, X, y, title):
	plt.scatter(X[np.argwhere(y==0).flatten(),0],X[np.argwhere(y==0).flatten(),1],s = 50, color = 'blue', edgecolor = 'k')
	plt.scatter(X[np.argwhere(y==1).flatten(),0],X[np.argwhere(y==1).flatten(),1],s = 50, color = 'red', edgecolor = 'k')

	plt.grid(False)
	plt.tick_params(
	axis='x',
	which='both',
	bottom='off',
	top='off')

	x_range = plt.xlim()
	y_range = plt.ylim()
	v1 = np.linspace(x_range[0],x_range[1],500)
	v2 = np.linspace(y_range[0],y_range[1],500)

	s,t = np.meshgrid(v1,v2)
	s = np.reshape(s,(np.size(s),1))
	t = np.reshape(t,(np.size(t),1))
	h = np.concatenate((s,t),1)

	z = model.predict(h)

	s.shape = (np.size(v1),np.size(v2))
	t.shape = (np.size(v1),np.size(v2))
	z.shape = (np.size(v1),np.size(v2))

	plt.contourf(s,t,z,colors = ['blue','red'],alpha = 0.2,levels = range(-1,2))
	if len(np.unique(z)) > 1:
	    plt.contour(s,t,z,colors = 'k', linewidths = 2)
	plt.title(title)
	plt.show()
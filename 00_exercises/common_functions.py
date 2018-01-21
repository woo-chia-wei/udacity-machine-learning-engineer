import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

def make_terrain_data(n_points=1000):
    random.seed(42)
    grade = [random.random() for ii in range(0,n_points)]
    bumpy = [random.random() for ii in range(0,n_points)]
    error = [random.random() for ii in range(0,n_points)]
    
    features = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    targets = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
    
    for ii in range(0, len(targets)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            targets[ii] = 1.0

    features = np.array(features)
    targets = np.array(targets)

    return features, targets

def plot_2d_scatter(X, y, colors=['blue', 'red']):
	plt.scatter(X[:,0], X[:,1], c=y, cmap=ListedColormap(colors)) 
	plt.show()
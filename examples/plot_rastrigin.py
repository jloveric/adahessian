"""
Modified from this gist
https://gist.github.com/miku/fca6afe05d65302f14c2b6f5242458d6
"""

#!/usr/bin/env python
# coding: utf-8

"""
https://en.wikipedia.org/wiki/Rastrigin_function
Non-convex function for testing optimization algorithms.
"""

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy as np

def rastrigin(X, Y, A:float=10):
    return A + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in [X,Y]])

if __name__ == '__main__':
    X = np.linspace(-4, 4, 200)    
    Y = np.linspace(-4, 4, 200)    

    X, Y = np.meshgrid(X, Y)

    Z = rastrigin(X, Y, A=10)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)  
    plt.show()  
    #plt.savefig('rastrigin.png')

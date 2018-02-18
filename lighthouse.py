import numpy as np
from numpy import pi, tan, log, exp
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def normalize(xs, ys):
    norm = sum(np.diff(xs) * ys[1:])
    return xs, ys/norm

class Game:

    def __init__(self, alpha, beta, N):
        self.alpha = alpha
        self.beta = beta
        self.N = N

    def plot(self, ax, x_range=(-10,10) ): 
        xdata = np.linspace(*x_range, 1000)
        ydata = self.L(xdata)
        
        xs, ys = normalize(xdata, exp(ydata) )

        ax.plot(xs, ys)
        ax.axvline(self.x_mean, ls='--', color='orange', label='mean')
        return fig 

    def flash(self):
        thetas = np.random.uniform(-pi/2, pi/2, size=N)
        xs = self.beta * tan(thetas) + self.alpha
        self.x_mean = xs.mean()
        def L(alphas):
            beta  = self.beta
            return np.array([-sum( log(beta**2+(xs-alpha)**2) ) for alpha in alphas])
        self.L = L

N = 100
alpha = 1
beta = 1
game = Game(alpha, beta, N)
game.flash()

fig, ax = plt.subplots() 
game.plot(ax)
ax.axvline(alpha, ls='--', color='red', label='true alpha')

ax.legend()
plt.show(fig)


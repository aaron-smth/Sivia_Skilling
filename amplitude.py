import numpy as np
from numpy import pi, tan, log, exp
from functools import partial
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.mlab import griddata


def normalize(arr, unit):
    norm = sum(arr*unit)
    return arr/norm


class Game:

    def __init__(self, A, B, n0, x0, omega, x_range):
        self.A = A
        self.B = B
        self.n0 = n0
        self.omega = omega
        self.x0 = x0
        self.x_range = x_range
        self.prepare()

    def prepare(self):
        def Dk(xk, A, B):
            n0,x0,omega = self.n0,self.x0,self.omega
            return n0 * (A*exp(-(xk-x0)**2/(2*omega**2)) + B )
        self.Dk = Dk

        def L(Ds, Ns):
            return sum( Ns * log(Ds) - Ds )  
        self.L = L

    def recording(self):
        Ds = self.Dk(self.x_range, self.A, self.B)
        Ns = np.random.poisson(lam=Ds)
        self.Ds, self.Ns = Ds, Ns
        self.L = partial(self.L, Ns=Ns)
    
        def pdf(As, Bs):
            def pack(A, B):
                Ds = self.Dk(self.x_range, A, B)
                return self.L( Ds )
            pack_v = np.vectorize(pack)
            return pack_v(As, Bs)
        self.pdf = pdf

    def plot(self, ax1, ax2 ): 
        x1 = np.linspace(self.x_range[0], self.x_range[-1], 1000)
        y1 = self.Dk(x1, self.A, self.B)
        ax1.plot(x1,y1, label='Dk')
        
        ax1.bar(self.x_range, self.Ns, align='center', label='experiment', alpha=0.7)
        
        npts = 200
        x = np.linspace(0.1,3, npts)
        y = np.linspace(0.1,3, npts)
        unit = (x[1] - x[0]) * (y[1] - y[0])
        xx,yy = np.meshgrid(x,y)
        LL = self.pdf(xx,yy)
        levels = [LL.max() * log(0.1*i) for i in range(1,10)]

        levels = np.linspace(LL.max()+log(0.1), LL.max()+log(0.9), 5)
        ax2.contour(xx,yy,LL,levels)
     
        return fig 

        

A = 1
B = 2
n0 = 35
x0 = 0
width = 12
omega = 2.12
x_range = np.arange(x0-width, x0+width+1)

game = Game(A,B,n0,x0,omega,x_range)
game.recording()


fig, axes = plt.subplots(1,2, figsize=(8,4) )
ax1, ax2 = axes
game.plot(ax1, ax2)

fig.legend()
plt.show(fig)

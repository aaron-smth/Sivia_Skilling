import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def normalize(f, x_range):
    norm = integrate.quad(f, 0, 1)[0]
    def f_new(x):
        return f(x) / norm
    return f_new

def x_mapper(a_range, b_range):
    a_diff = a_range[1] - a_range[0]
    b_diff = b_range[1] - b_range[0]
    def x_map(x):
        percent = (x - b_range[0]) / b_diff
        return a_range[0] + a_diff * percent
    return x_map

def prob_hash(a_range, b_range=(0,1)):
    def hash_dec(f):
        mapper = x_mapper(a_range, b_range)
        def f_new(x):
            new_x = mapper(x)
            return f(new_x)
        return normalize(f_new, b_range)
    return hash_dec


@prob_hash((-1,1))
def uniform(x):
    if type(x) in [int, float]:
        return 1
    else:
        return np.ones(len(x))

@prob_hash((-1,1))
def centered(x):
    return 1-np.cosh(x)

@prob_hash((-1,1))
def spread(x):
    return np.cosh(x) 

class Game:

    def __init__(self, p, N, prior):
        self.p = p
        self.prior = prior
        self.pdf = prior
        self.N = N

    def plot(self, ax, x_range=(0,1) ): 
        xdata = np.linspace(*x_range, 1000)
        ydata = self.pdf(xdata)
        
        ax.plot(xdata,ydata)

    def toss(self):
        ups = np.random.choice([0,1], p=[1-self.p,self.p], size=self.N)
        up = sum(ups)
        def new_pdf(h):
           p = h**up*(1-h)**(self.N-up) * self.prior(h)
           return p
        self.pdf = normalize(new_pdf,(0,1))


N = 100
p = 0.3

game1 = Game(p=p, N=N, prior=uniform)
game2 = Game(p=p, N=N, prior=centered)
game3 = Game(p=p, N=N, prior=spread)

game1.toss()
game2.toss()
game3.toss()

fig, ax = plt.subplots() 
game1.plot(ax)
game2.plot(ax)
game3.plot(ax)
ax.axvline(p, '--')

plt.show(fig)





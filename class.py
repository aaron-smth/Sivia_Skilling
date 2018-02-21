from functools import partial
from numpy import pi, tan, log, exp

def normalize(arr, unit):
    norm = sum(arr*unit)
    return arr/norm

class ParameterSapce:

    def __init__(self, **kwargs):
        self.data = kwargs['data']
        self.L = kwargs['L']
        self.parameters = {k:v for k,v in kwargs if k not in ['L','data']} 
        self.unit = 0.01
        self._prepare()

    def _prepare(self):
        self.known = {k:v for k,v in self.parameters if type(v) in [int, float]}
        self.unkown = {k:v for k,v in self.parameters if k not in self.known}

        self.L = partial(self.L, **self.known)
        def to_prob(L): 
            L -= L.max() ## to avoid infinity 
            prob = normalize( exp(L) , self.unit)
            return prob  
        self.to_pdf = to_pdf
   
     

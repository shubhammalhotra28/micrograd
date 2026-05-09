import random
from nanograd.engine import Value

class Module:
    
    def zero_grad(self):
        for p in self.parameter():
            p.grad = 0
        
    def parameters(self):
        return []
    

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
    
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w,x)), self.b)
        return act.relu() if self.nonlin else act
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"





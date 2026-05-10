import random
import math
from nanograd.engine import Value
from nanograd.ops import softmax, cross_entropy

class Tensor:

    def __init__(self, data):
        # accepts 2d list of floats or Value objects
        self.data = [
            [v if isinstance(v, Value) else Value(v) for v in row] for row in data]
    
    @property
    def shape(self):
        return (len(self.data), len(self.data[0]))
    
    def parameters(self):
        return [v for row in self.data for v in row]
    
    @classmethod
    def randn(cls, rows, cols, std=0.1):
        data = [[random.gauss(0, std) for _ in range(cols)] for _ in range(rows)]
        return cls(data)
    
    def __matmul__(self, other):
        m, k1 = self.shape
        k2, n = other.shape

        assert k1 == k2, "Matrix dimensions must match"

        out = [[None for j in range(n)] for i in range(m)]

        for i in range(m):
            for j in range(n):
                total = Value(0.0)
                for k in range(k1):
                    total += self.data[i][k] * other.data[k][j]
                
                out[i][j] = total
        
        return Tensor(out)

    
    @property
    def T(self):
        rows, cols = self.shape
        return Tensor([
            [self.data[i][j] for i in range(rows)] for j in range(cols)
        ])
    
    def __truediv__(self, divisor):
        return Tensor([
            [v / divisor for v in row] for row in self.data
        ])
    
    def softmax(self, axis=-1):
        return Tensor([softmax(row) for row in self.data])
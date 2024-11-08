import numpy
import math
from micrograd import draw_dot
from IPython.display import display
learning_rate = 0.01

class Value:
    def __init__(self, data, _children=(), op='', label=''):
        self.data = data
        self._backward = lambda: None
        self._prev = set(_children)
        self.grad = 0.0
        self._op = op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):    
        out = Value(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = -1.0 * out.grad
        out._backward = _backward
        return out 
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad = other.data * out.grad 
            other.grad = self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = Value(self.data / other.data, (self, other), '/')
        def _backward():
            self.grad = 1.0 / other.data * out.grad
            other.grad = -self.data / (other.data ** 2) * out.grad
        out._backward = _backward
        return out

    def square(self):
        out = Value(self.data ** 2, (self,), '**2')
        def _backward():
            self.grad = 2 * self.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad = (1.0 / self.data) * out.grad
        out._backward = _backward
        return out
    
    # Sample activations as placeholders
    def exp(self):
        return None

    def tanh(self):
        return None

    def reLU(self):
        return None

    def softmax(self):
        return None




# Constructing a computational graph
x = Value(2, label='x')
y = Value(1, label='y')
z = Value(1, label='z')

xAddy = x + y; xAddy.label = 'xAddy'
xAddy2 = xAddy.square(); xAddy2.label = 'xAddy2'
xAddy2Addz = xAddy2 + z; xAddy2Addz.label = 'xAddy2Addz'
logxAddy2Addz = xAddy2Addz.log(); logxAddy2Addz.label = 'logxAddy2Addz'

# Perform backward pass
logxAddy2Addz.grad = 1.0
# last._backward()
# xAddy2Addz._backward()
# xAddy2._backward()
# xAddy._backward()


topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(logxAddy2Addz)

for node in reversed(topo):
    node._backward()
draw_dot(logxAddy2Addz).render(view=True)
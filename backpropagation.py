import numpy
from micrograd import draw_dot
from IPython.display import display
learning_rate = 0.01


class Value:
    def __init__(self, data, _children=(), op='', label=''):
        self.data = data
        self._backward = lambda:None
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
        out = Value(self.data - other.data, (self, other), '+')
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = -1.0 * out.grad
        
        out._backward = _backward
        return out 
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '+')
        def _backward():
            self.grad = other.data * out.grad 
            other.grad = self.data * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = Value(self.data / other.data, (self, other), '+')
        def _backward():
            self.grad = 1.0 / other.data * out.grad
            other.grad = self.data * (-1.0) / other.data**2 * out.grad

        out._backward = _backward
        return out
    
    # activations
    # tanh function
    def tanh():
        return None


    # softmax function
    def softmax():
        return None


a = Value(2.0, label = 'a')
b = Value(-3.0, label = 'b')
c = Value(10.0, label ='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value (-2.0, label = 'f')
L = d * f; L.label = 'L'

L.grad = 1.0
L._backward()
d._backward()
e._backward()
draw_dot(L).render(view=True)
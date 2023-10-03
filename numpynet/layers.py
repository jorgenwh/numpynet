import numpy as np
from typing import Tuple
from math import sqrt
from .gate import Gate

class Layer(Gate):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("__call__ has not been implemented for this layer object.")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward has not been implemented for this layer object.")

    def backward(self, *args, **kwargs):
        raise NotImplementedError("backward has not been implemented for this layer object.")

    def update_step(self, lr: float):
        raise NotImplementedError("update_step has not been implemented for this layer object.")

    def zero_grad(self):
        raise NotImplementedError("zero_grad has not been implemented for this layer object.")

    def get_parameters(self):
        raise NotImplementedError("get_parameters has not been implemented for this layer object.")


class Flatten(Layer):
    def __init__(self):
        self.X_shape = None

    def __call__(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
        return self.forward(X, grad)

    def forward(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
        z = X.reshape(X.shape[0], -1)
        if grad:
            self.X_shape = X.shape
        return z 

    def backward(self, dY: np.ndarray) -> np.ndarray:
        assert self.X_shape is not None
        dX = dY.reshape(self.X_shape)
        return dX

    def update_step(self, lr: float) -> None:
        pass

    def zero_grad(self) -> None:
        pass

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array([]), np.array([]))

    
class Linear(Layer):
    def __init__(self, in_size: int, out_size: int):
        self.in_size = in_size
        self.out_size = out_size
        
        # Initialize the parameters
        self.W = np.random.randn(in_size, out_size) / sqrt(in_size)
        self.b = np.random.randn(1, out_size) / sqrt(in_size)

        # Local variables to cache necessary data
        self.X = None
        self.output = None
        self.gradW = np.zeros((in_size, out_size))
        self.gradb = np.zeros((1, out_size))

    def __call__(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
        return self.forward(X, grad)

    def forward(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
        z = np.dot(X, self.W) + self.b
        if grad:
            self.X = X
            self.output = z
        return z

    def backward(self, dY: np.ndarray) -> np.ndarray:
        assert self.X is not None
        self.gradW = np.dot(self.X.T, dY)
        self.gradb = np.sum(dY, axis=0, keepdims=True)
        dX = np.dot(dY, self.W.T)
        return dX

    def update_step(self, lr: float) -> None:
        self.W -= (lr * self.gradW)
        self.b -= (lr * self.gradb)

    def zero_grad(self) -> None:
        self.gradW *= 0
        self.gradb *= 0

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.W, self.b)


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize parameters
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) / sqrt(in_channels)

        # Local variables to cache necessary data
        self.X = None
        self.output = None
        self.gradW = np.zeros_like(self.W)

    def __call__(self, x, grad=True):
        return self.forward(x)

    def forward(self, x: np.ndarray, grad: bool = True) -> np.ndarray:
        B, Cin, Hin, Win = x.shape
        Cout = self.out_channels
        Hout = ((Hin - self.kernel_size + 2*self.padding) // self.stride) + 1
        Wout = ((Win - self.kernel_size + 2*self.padding) // self.stride) + 1
        Ck, Hk, Wk = self.W.shape[1:]
        stride = self.stride

        z = np.zeros((B, Cout, Hout, Wout))

        for b in range(B):
            X = x[b]
            for cout in range(Cout):
                kernel = self.W[cout]

                for c in range(Cin):
                    for h in range(Hout):
                        for w in range(Wout):
                            for i in range(Hk):
                                for j in range(Wk):
                                    z[b,cout,h,w] += X[c,h*stride + i,w*stride + j] * kernel[c,i,j]

        if grad:
            self.X = X
            self.output = z

        return z

    def backward(self, dY: np.ndarray) -> np.ndarray:
        assert self.X is not None
        B, Cout, Hout, Wout = dY.shape

        dX = np.zeros_like(self.X)
        self.gradW = np.zeros_like(self.W)

        return dX

    def update_step(self, lr: float) -> None:
        self.W -= (lr * self.gradW)

    def zero_grad(self) -> None:
        self.gradW *= 0

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.W, None)

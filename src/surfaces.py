import numpy as np

class Surface:
    def height(self, x, y):
        raise NotImplementedError

    def gradient(self, x, y, h=1e-5):
        """
        Computes the gradient (dz/dx, dz/dy) using central finite differences.
        """
        dz_dx = (self.height(x + h, y) - self.height(x - h, y)) / (2 * h)
        dz_dy = (self.height(x, y + h) - self.height(x, y - h)) / (2 * h)
        return np.array([dz_dx, dz_dy])

    def hessian(self, x, y, h=1e-5):
        """
        Computes the Hessian matrix using central finite differences.
        [[d2z/dx2, d2z/dxdy],
         [d2z/dydx, d2z/dy2]]
        """
        f = self.height
        d2z_dx2 = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h**2)
        d2z_dy2 = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h**2)
        d2z_dxdy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)
        
        return np.array([[d2z_dx2, d2z_dxdy],
                         [d2z_dxdy, d2z_dy2]])

class FlatSurface(Surface):
    def height(self, x, y):
        return np.zeros_like(x)

class ScenarioA(Surface):
    def __init__(self):
        self.centers = np.array([
            [1.0, -0.5],
            [1.5, 0.4],
            [2.25, -0.6]
        ])
        self.sigma2 = 0.1
        self.amplitude = 0.1

    def height(self, x, y):
        z = 0
        for x0, y0 in self.centers:
            z += np.exp(-((x - x0)**2 + (y - y0)**2) / self.sigma2)
        return self.amplitude * z

class ScenarioB(Surface):
    def height(self, x, y):
        term1 = (1 - np.tanh((x - 1.5) / 0.05)) * (1 + np.tanh((y - 1) / 0.05))
        term2 = 0.1 * np.tanh((x - y - 2) / 0.6)
        return term1 + term2

class ScenarioC(Surface):
    def height(self, x, y):
        term1 = (1 + np.tanh((x - 3.75) / 0.05)) * (1 - np.tanh((y - 2.75) / 0.05))
        term2 = (1 - np.tanh((x - 1.5) / 0.05)) * (1 + np.tanh((y - 1) / 0.05))
        term3 = 0.05 * (1 + np.tanh((x - y - 2) / 0.6)) * (1 - np.tanh((x - 3.75) / 0.05))
        term4 = 0.1 * np.tanh((y - 4.25) / 0.6)
        return term1 + term2 + term3 + term4

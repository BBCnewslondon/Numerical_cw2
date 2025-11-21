import numpy as np

class PhysicsEngine:
    def __init__(self, surface, g=9.81, mu=0.075, m=0.045, tol=1e-6):
        self.surface = surface
        self.g = g
        self.mu = mu
        self.m = m
        self.tol = tol

    def get_normal(self, x, y):
        grad = self.surface.gradient(x, y)
        # n = (-dz/dx, -dz/dy, 1)
        return np.array([-grad[0], -grad[1], 1.0])

    def get_curvature_matrix(self, x, y):
        hess = self.surface.hessian(x, y)
        # J = [[-z_xx, -z_xy, 0], [-z_xy, -z_yy, 0], [0, 0, 0]]
        J = np.zeros((3, 3))
        J[0, 0] = -hess[0, 0]
        J[0, 1] = -hess[0, 1]
        J[1, 0] = -hess[1, 0]
        J[1, 1] = -hess[1, 1]
        return J

    def equations_of_motion(self, t, state):
        """
        Computes the derivatives of the state vector [x, y, vx, vy].
        """
        x, y, vx, vy = state
        
        # 1. Geometry
        grad = self.surface.gradient(x, y)
        n = np.array([-grad[0], -grad[1], 1.0])
        n_norm_sq = np.dot(n, n)
        n_norm = np.sqrt(n_norm_sq)
        
        J = self.get_curvature_matrix(x, y)
        
        # 2. Velocity in 3D
        # v . n = 0 => -vx*zx - vy*zy + vz = 0 => vz = vx*zx + vy*zy
        vz = vx * grad[0] + vy * grad[1]
        v = np.array([vx, vy, vz])
        v_norm = np.linalg.norm(v)
        
        # 3. Forces
        # Normal Force N
        # N = -m * (v.T * J * v) / ||n||^2 * n
        v_J_v = np.dot(v, np.dot(J, v))
        N = -self.m * v_J_v / n_norm_sq * n
        
        # Tangential Gravity T
        # T = -mg * (ez - (n . ez) / ||n||^2 * n)
        ez = np.array([0, 0, 1])
        n_dot_ez = n[2] # Since ez is (0,0,1)
        T = -self.m * self.g * (ez - (n_dot_ez / n_norm_sq) * n)
        
        # Friction F
        # F = -mu * m * g * ||n|| / ||v|| * v
        if v_norm < self.tol:
            F = np.zeros(3)
        else:
            F = -self.mu * self.m * self.g * (n_norm / v_norm) * v
            
        # Total Force
        TotalForce = N + T + F
        
        # Acceleration
        a = TotalForce / self.m
        
        # Return derivatives [vx, vy, ax, ay]
        return [vx, vy, a[0], a[1]]

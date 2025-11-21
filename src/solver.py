import numpy as np
from scipy.optimize import root, minimize
from scipy.integrate import solve_ivp

class ShootingSolver:
    def __init__(self, physics_engine):
        self.physics = physics_engine

    def solve(self, objective_function, initial_guess, method='root'):
        """
        Solves the BVP using the shooting method.
        
        Args:
            objective_function: A function that takes params and returns residuals (for root) or scalar cost (for minimize).
            initial_guess: Initial guess for the parameters.
            method: 'root' or 'minimize'.
        """
        if method == 'root':
            sol = root(objective_function, initial_guess, method='hybr')
            return sol
        elif method == 'minimize':
            sol = minimize(objective_function, initial_guess, method='Nelder-Mead')
            return sol
        else:
            raise ValueError("Method must be 'root' or 'minimize'")

    def integrate(self, state0, t_span):
        return solve_ivp(
            self.physics.equations_of_motion,
            t_span,
            state0,
            rtol=1e-8,
            atol=1e-8
        )

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def run_simulation(physics_engine, initial_state, t_span, t_eval=None, events=None):
    """
    Runs the simulation for a given initial state.
    """
    sol = solve_ivp(
        physics_engine.equations_of_motion,
        t_span,
        initial_state,
        t_eval=t_eval,
        events=events,
        rtol=1e-8,
        atol=1e-8
    )
    return sol

def plot_trajectory(surface, sol, title="Trajectory"):
    """
    Plots the trajectory on the surface contour.
    """
    x = sol.y[0]
    y = sol.y[1]
    
    # Create grid for contour plot
    x_range = np.linspace(min(x.min(), -1), max(x.max(), 5), 100)
    y_range = np.linspace(min(y.min(), -1), max(y.max(), 5), 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = surface.height(X[i, j], Y[i, j])
            
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Height z')
    plt.plot(x, y, 'r-', label='Path')
    plt.plot(x[0], y[0], 'go', label='Start')
    plt.plot(x[-1], y[-1], 'bo', label='End')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def calculate_energy(physics_engine, sol):
    """
    Calculates the total energy (KE + PE) along the trajectory.
    """
    m = physics_engine.m
    g = physics_engine.g
    
    energies = []
    for i in range(len(sol.t)):
        x, y, vx, vy = sol.y[:, i]
        z = physics_engine.surface.height(x, y)
        grad = physics_engine.surface.gradient(x, y)
        vz = vx * grad[0] + vy * grad[1]
        v_sq = vx**2 + vy**2 + vz**2
        
        KE = 0.5 * m * v_sq
        PE = m * g * z
        energies.append(KE + PE)
        
    return np.array(energies)

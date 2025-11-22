import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import least_squares, minimize
from scipy.integrate import solve_ivp

# Ensure src can be imported when running this script from the repository root
sys.path.append(os.path.abspath("src"))

from surfaces import ScenarioA, ScenarioB, ScenarioC
from physics import PhysicsEngine
from solver import ShootingSolver
from utils import calculate_energy


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def save_terrain_plot(surface, filename, x_range, y_range, title="Terrain", markers=None, levels=40):
    """Creates and saves a contour plot of the terrain for the given surface.

    markers: list of tuples (x, y, kwargs dict) to plot markers on the map.
    """
    X, Y = np.meshgrid(x_range, y_range)
    Z = surface.height(X, Y)

    fig, ax = plt.subplots(figsize=(9, 7))
    cs = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="Height z")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if markers:
        for (mx, my, mk) in markers:
            ax.plot(mx, my, **mk)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def save_trajectory_plot(surface, sol, filename, title="Trajectory"):
    x = sol.y[0]
    y = sol.y[1]

    # Create grid for contour plot with reasonable extents
    x_min, x_max = min(x.min(), -1), max(x.max(), 5)
    y_min, y_max = min(y.min(), -1), max(y.max(), 5)
    x_range = np.linspace(x_min, x_max, 150)
    y_range = np.linspace(y_min, y_max, 150)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = surface.height(X[i, j], Y[i, j])

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.6)
    fig.colorbar(cs, ax=ax, label="Height z")
    ax.plot(x, y, "r-", label="Path")
    ax.plot(x[0], y[0], "go", label="Start")
    ax.plot(x[-1], y[-1], "bo", label="End")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def create_plots_for_A(outdir, run_full=True):
    """Recreate and save plots for Scenario A."""
    mkdir_p(outdir)
    surface = ScenarioA()
    physics = PhysicsEngine(surface)
    solver = ShootingSolver(physics)

    # Solve using same objective as notebook (root solver)
    target_pos = np.array([3.5, 0.0])
    target_speed = 0.1

    def objective(params):
        vx0, vy0, T = params
        if T <= 0:
            return [100, 100, 100]
        initial_state = [0, 0, vx0, vy0]
        sol = solver.integrate(initial_state, [0, T])
        xf, yf, vxf, vyf = sol.y[:, -1]
        grad = surface.gradient(xf, yf)
        vzf = vxf * grad[0] + vyf * grad[1]
        vf = np.sqrt(vxf**2 + vyf**2 + vzf**2)
        return [xf - target_pos[0], yf - target_pos[1], vf - target_speed]

    # Use a reasonable guess
    guess = [2.0, 0.5, 3.0]
    sol_root = None
    if run_full:
        sol_root = solver.solve(objective, guess)

    if sol_root and sol_root.success and run_full:
        vx0, vy0, T = sol_root.x
        initial_state = [0, 0, vx0, vy0]
        sol_path = solver.integrate(initial_state, [0, T])
    else:
        # fallback simple ballistic sample path for plotting
        initial_state = [0, 0, guess[0], guess[1]]
        sol_path = solver.integrate(initial_state, [0, guess[2]])

    filename = os.path.join(outdir, "scenarioA_trajectory.png")
    save_trajectory_plot(surface, sol_path, filename, title="Scenario A Solution")
    print(f"Saved Scenario A trajectory to {filename}")


def create_plots_for_B(outdir, run_full=True):
    mkdir_p(outdir)
    surface = ScenarioB()
    physics = PhysicsEngine(surface)
    solver = ShootingSolver(physics)

    # Terrain plot
    x_range = np.linspace(-1, 5, 250)
    y_range = np.linspace(-1, 5, 250)
    markers = [
        (0, 0, {"marker": "o", "color": "g", "label": "Start"}),
        (2.6, 1.0, {"marker": "*", "color": "r", "label": "Midpoint", "markersize": 10}),
        (2.5, 2.5, {"marker": "x", "color": "b", "label": "Endpoint", "markersize": 8}),
    ]
    filename = os.path.join(outdir, "scenarioB_terrain.png")
    save_terrain_plot(surface, filename, x_range, y_range, title="Scenario B: Terrain before shot", markers=markers, levels=40)
    print(f"Saved Scenario B terrain to {filename}")

    if not run_full:
        # create fallback sample trajectory using a plausible initial state
        initial_state = [0, 0, 2.5, 1.0]
        sol_path = solver.integrate(initial_state, [0, 3.0])
        filename = os.path.join(outdir, "scenarioB_trajectory.png")
        save_trajectory_plot(surface, sol_path, filename, title="Scenario B sample trajectory")
        print(f"Saved Scenario B fallback trajectory to {filename}")
        return

    # Recreate the robust objective and run the least squares optimization
    midpoint = np.array([2.6, 1.0])
    endpoint = np.array([2.5, 2.5])

    def ball_stopped(t, y):
        vx, vy = y[2], y[3]
        speed = np.sqrt(vx**2 + vy**2)
        return speed - 1e-3
    ball_stopped.terminal = True
    ball_stopped.direction = -1

    def objective_robust(params):
        vx0, vy0, T1, T2 = params
        if T1 <= 0.1 or T2 <= T1 + 0.1:
            return np.ones(4) * 100.0
        initial_state = [0, 0, vx0, vy0]
        sol1 = solve_ivp(physics.equations_of_motion, [0, T1], initial_state, rtol=1e-6, atol=1e-6)
        if not sol1.success:
            return np.ones(4) * 100.0
        state_mid = sol1.y[:, -1]
        sol2 = solve_ivp(physics.equations_of_motion, [T1, T2], state_mid, rtol=1e-6, atol=1e-6, events=ball_stopped)
        if not sol2.success and sol2.status != 1:
            return np.ones(4) * 100.0
        state_end = sol2.y[:, -1]
        return [state_mid[0] - midpoint[0], state_mid[1] - midpoint[1], state_end[0] - endpoint[0], state_end[1] - endpoint[1]]

    guess = [2.5, 1.0, 1.5, 3.0]
    lower_bounds = [-np.inf, -np.inf, 0.1, 0.1]
    upper_bounds = [np.inf, np.inf, 10.0, 10.0]

    res = least_squares(objective_robust, guess, bounds=(lower_bounds, upper_bounds), method='trf')
    print("Scenario B least_squares success:", res.success)
    if res.success:
        vx0, vy0, T1, T2 = res.x
        initial_state = [0, 0, vx0, vy0]
        sol_final = solve_ivp(physics.equations_of_motion, [0, T2], initial_state, t_eval=np.linspace(0, T2, 500), rtol=1e-8, atol=1e-8)
    else:
        # fallback
        initial_state = [0, 0, guess[0], guess[1]]
        sol_final = solve_ivp(physics.equations_of_motion, [0, guess[3]], initial_state, t_eval=np.linspace(0, guess[3], 500), rtol=1e-8, atol=1e-8)

    filename = os.path.join(outdir, "scenarioB_trajectory.png")
    save_trajectory_plot(surface, sol_final, filename, title="Scenario B Solution")
    print(f"Saved Scenario B trajectory to {filename}")


def create_plots_for_C(outdir, run_full=True):
    mkdir_p(outdir)
    surface = ScenarioC()
    physics = PhysicsEngine(surface)
    solver = ShootingSolver(physics)

    # Terrain plot
    x_range = np.linspace(-1, 6, 300)
    y_range = np.linspace(-1, 6, 300)
    markers = [
        (0, 0, {"marker": "o", "color": "g", "label": "Start"}),
        (2.6, 1.0, {"marker": "*", "color": "r", "label": "Midpoint Shot 1", "markersize": 10}),
        (4.0, 3.8, {"marker": "*", "color": "c", "label": "Midpoint Shot 2", "markersize": 10}),
        (4.5, 3.4, {"marker": "x", "color": "b", "label": "Endpoint", "markersize": 8}),
    ]
    filename = os.path.join(outdir, "scenarioC_terrain.png")
    save_terrain_plot(surface, filename, x_range, y_range, title="Scenario C: Terrain before shots", markers=markers, levels=50)
    print(f"Saved Scenario C terrain to {filename}")

    if not run_full:
        # fallback sample shots
        sol1 = solve_ivp(physics.equations_of_motion, [0, 3.0], [0, 0, 2.0, 1.0], t_eval=np.linspace(0, 3.0, 200), rtol=1e-8, atol=1e-8)
        filename = os.path.join(outdir, "scenarioC_shot1.png")
        save_trajectory_plot(surface, sol1, filename, title="Scenario C: Shot 1 Sample")
        sol2 = solve_ivp(physics.equations_of_motion, [0, 3.0], [sol1.y[0, -1], sol1.y[1, -1], 1.0, 0.5], t_eval=np.linspace(0, 3.0, 200), rtol=1e-8, atol=1e-8)
        filename = os.path.join(outdir, "scenarioC_shot2.png")
        save_trajectory_plot(surface, sol2, filename, title="Scenario C: Shot 2 Sample")
        print(f"Saved Scenario C fallback shot1/shot2 to {outdir}")
        return

    # Shot 1: find an initial velocity that stops near target midpoint
    target_midpoint = np.array([2.6, 1.0])
    def ball_stopped(t, y):
        vx, vy = y[2], y[3]
        speed = np.sqrt(vx**2 + vy**2)
        return speed - 1e-3
    ball_stopped.terminal = True
    ball_stopped.direction = -1

    def shot1_objective(params):
        vx0, vy0 = params
        initial_state = [0, 0, vx0, vy0]
        sol = solve_ivp(physics.equations_of_motion, [0, 10], initial_state, events=ball_stopped, rtol=1e-6, atol=1e-6, dense_output=True)
        if sol.status == -1:
            return 100.0
        t_eval = np.linspace(0, sol.t[-1], 100)
        path = sol.sol(t_eval)
        positions = path[:2, :].T
        dists = np.linalg.norm(positions - target_midpoint, axis=1)
        min_dist = np.min(dists)
        stop_penalty = 0.0 if sol.status == 1 else 10.0
        return min_dist + stop_penalty

    guess1 = [2.0, 1.0]
    res1 = minimize(shot1_objective, guess1, method='Nelder-Mead', tol=1e-3)
    print("Shot 1 optimization success:", res1.success)
    if res1.success:
        vx_opt, vy_opt = res1.x
        sol1 = solve_ivp(physics.equations_of_motion, [0, 10], [0, 0, vx_opt, vy_opt], events=ball_stopped, rtol=1e-6, atol=1e-6, dense_output=True)
    else:
        sol1 = solve_ivp(physics.equations_of_motion, [0, 3.0], [0, 0, guess1[0], guess1[1]], t_eval=np.linspace(0, 3.0, 200))

    filename = os.path.join(outdir, "scenarioC_shot1.png")
    save_trajectory_plot(surface, sol1, filename, title="Scenario C: Shot 1")
    print(f"Saved Scenario C shot 1 to {filename}")

    # If shot 1 solved and ended in stop, use the stopping position; otherwise use the last point
    if sol1.status == 1:
        stop_pos = sol1.y[:2, -1]
    else:
        stop_pos = sol1.y[:2, -1]

    # Shot 2: use the same solver pattern as notebook
    midpoint_2 = np.array([4.0, 3.8])
    endpoint_2 = np.array([4.5, 3.4])

    def objective_shot2(params):
        vx0, vy0, T1, T2 = params
        if T1 <= 0.1 or T2 <= T1 + 0.1:
            return [100, 100, 100, 100]
        initial_state = [stop_pos[0], stop_pos[1], vx0, vy0]
        sol = solve_ivp(physics.equations_of_motion, [0, T2], initial_state, rtol=1e-8, atol=1e-8, dense_output=True, events=ball_stopped)
        if not sol.success and sol.status != 1:
            return [100, 100, 100, 100]
        try:
            state_T1 = sol.sol(T1)
        except Exception:
            state_T1 = sol.y[:, -1]
        if sol.status == 1:
            state_T2 = sol.y[:, -1]
        else:
            state_T2 = sol.sol(T2)
        return [state_T1[0] - midpoint_2[0], state_T1[1] - midpoint_2[1], state_T2[0] - endpoint_2[0], state_T2[1] - endpoint_2[1]]

    guess2 = [1.0, 1.0, 1.0, 2.0]
    sol2 = solver.solve(objective_shot2, guess2)
    print("Shot 2 solver success:", sol2.success)
    if sol2.success:
        vx0, vy0, T1, T2 = sol2.x
        initial_state = [stop_pos[0], stop_pos[1], vx0, vy0]
        sol_path2 = solver.integrate(initial_state, [0, T2])
    else:
        # fallback
        initial_state = [stop_pos[0], stop_pos[1], guess2[0], guess2[1]]
        sol_path2 = solver.integrate(initial_state, [0, guess2[3]])

    filename = os.path.join(outdir, "scenarioC_shot2.png")
    save_trajectory_plot(surface, sol_path2, filename, title="Scenario C: Shot 2")
    print(f"Saved Scenario C shot 2 to {filename}")


def create_all_plots(outdir="plots", run_full=True):
    mkdir_p(outdir)
    create_plots_for_A(outdir, run_full=run_full)
    create_plots_for_B(outdir, run_full=run_full)
    create_plots_for_C(outdir, run_full=run_full)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create and save plots for all scenarios.")
    parser.add_argument("--outdir", default="plots", help="Output directory for plots")
    parser.add_argument("--no-full", dest="run_full", action="store_false", help="Don't run full, expensive optimizations (use quick sample runs)")
    args = parser.parse_args()
    create_all_plots(outdir=args.outdir, run_full=args.run_full)

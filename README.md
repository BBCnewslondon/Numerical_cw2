# Crazy Golf BVP Solver

This project implements a numerical solver for the "Crazy Golf" problem, modeling a ball moving on a curved surface under gravity and friction.

## Project Structure

- `src/`: Source code for the physics engine, surface definitions, and solver.
  - `physics.py`: Equations of motion and force calculations.
  - `surfaces.py`: Definitions of the surface functions $z(x,y)$ for Scenarios A, B, and C.
  - `solver.py`: BVP solver implementation using shooting methods.
  - `utils.py`: Helper functions.
- `tests/`: Unit tests.
- `notebooks/`: Jupyter notebooks for running scenarios and analysis.
  - `Scenario_A.ipynb`: The Triple Mound.
  - `Scenario_B.ipynb`: The Obstacle Course.
  - `Scenario_C.ipynb`: The Two-Putt Challenge.
  - `Testing_and_Validation.ipynb`: Validation against known cases.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the notebooks in the `notebooks/` directory to see the solutions for each scenario.

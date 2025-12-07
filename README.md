# Crazy Golf BVP Solver

This project implements a numerical solver for the "Crazy Golf" problem, modeling a ball moving on a curved surface under gravity and friction.

## Project Structure

- `cw.py`: Main script to run the solver and generate plots.
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
- `plots/`: Directory for generated plots.

## Prerequisites

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the notebooks in the `notebooks/` directory to see the solutions for each scenario.

### Running the Main Script

To run the solver and generate plots, execute:

```bash
python cw.py
```

This will solve the scenarios and save plots to the `plots/` directory.

### Testing

Run the unit tests:

```bash
python -m pytest tests/
```

Or using unittest:

```bash
python -m unittest tests/test_cw.py
```

"""
CelerisAi is a Python-Taichi-based software designed for nearshore wave modeling. This solver offers high-performance simulations on various hardware platforms and seamlessly integrates with machine learning and artificial intelligence environments. The solver leverages the flexibility of Python for customization and interoperability, while Taichi's high-performance parallel programming capabilities ensure efficient computations.

This package contains modules for running domain-specific simulations, solving
problems, and providing general utilities:

- **domain.py**: Contains classes and functions defining the problem domain.
- **runner.py**: Manages the execution flow, including initialization and orchestration.
- **solver.py**: Implements the core solver logic for the defined domain and mathematical models.
- **utils.py**: A collection of helper utilities used across the package.

Example:

    >>> from celeris import domain, runner, solver, utils
    >>> domain_obj = domain.Domain(params={...})
    >>> solver_obj = solver.Solver(domain=domain_obj)
    >>> runner_obj = runner.Evolve(solver = solver_obj)
    >>> runner_obj.Evolve_Headless()
Attributes:

    __version__ (str): The current version of the CelerisAi.
    
"""

__version__ = "0.0.1"


from .domain import *

from .runner import *

from .solver import *

from .utils import *

__all__ = [
    "Domain",
    "Runner",
    "Solver",
    "Utils",
    "__version__",
]


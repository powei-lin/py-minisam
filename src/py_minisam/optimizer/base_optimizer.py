from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict

import numpy as np

from py_minisam.factor_graph import FactorGraph
from py_minisam.problem import Problem


class SolverStatus(Enum):
    Running = auto()
    GradientTooSmall = auto()  # eps > max(J'*f(x))
    RelativeStepSizeTooSmall = auto()  # eps > ||dx|| / ||x||
    ErrorTooSmall = auto()  # eps > ||f(x)||
    HitMaxIterations = auto()


@dataclass
class SolverParameters:
    gradient_threshold: float = 1e-16
    relative_step_threshold: float = 1e-16
    error_threshold: float = 1e-16
    initial_scale_factor: float = 1e-3
    max_iterations: int = 100


@dataclass
class ProblemResult:
    error_magnitude: float = 0.0
    gradient_magnitude: float = 0.0
    num_failed_linear_solves: int = 0
    iterations: int = 0
    status: SolverStatus = SolverStatus.Running


class BaseOptimizer:
    @abstractmethod
    def optimize(self, problem: Problem, max_iteration: int = 100):
        pass

    def optimize_factor_graph(
        self, factor_graph: FactorGraph, init_values: Dict[str, np.ndarray], max_iteration: int = 100
    ):
        problem = Problem.from_factor_graph(factor_graph, init_values)
        self.optimize(problem, max_iteration)

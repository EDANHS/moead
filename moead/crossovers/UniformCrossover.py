import numpy as np

from moead.crossovers import Crossover
from moead.solutions import Solution


class UniformCrossover(Crossover):
    """
    Uniform crossover: para cada gen, elige aleatoriamente del padre1 o padre2.
    Devuelve una instancia del mismo tipo de `parent1`.
    """
    def __init__(self, prob_cross: float = 0.9):
        self.prob_cross = prob_cross

    def execute(self, parent1: Solution, parent2: Solution, bounds: list) -> Solution:
        p1 = parent1.variables
        p2 = parent2.variables

        if np.random.rand() > self.prob_cross:
            return type(parent1)(np.copy(p1), parent1.objectives.shape[0], parent1.constraints.shape[0])

        mask = np.random.rand(*p1.shape) < 0.5
        child_vars = np.where(mask, p1, p2)

        # Apply clipping to bounds
        min_b = np.array([b[0] for b in bounds])
        max_b = np.array([b[1] for b in bounds])
        child_vars = np.clip(child_vars, min_b, max_b)

        return type(parent1)(child_vars, parent1.objectives.shape[0], parent1.constraints.shape[0])

import numpy as np

from ..crossovers import Crossover
from .EvolutionaryOperator import EvolutionaryOperator
from ..mutations import Mutation
from ..problems.Problem import Problem
from ..solutions import Solution


class CrossoverMutation(EvolutionaryOperator):
    """
    Implementa la reproducción usando SBX + Mutación Polinómica
    y selección de padres híbrida.
    """
    def __init__(self, 
                 crossover: Crossover, 
                 mutation: Mutation, 
                 mating_prob: float = 0.9): 
        self.crossover_op = crossover
        self.mutation_op = mutation
        self.mating_prob = mating_prob

    def execute(self, 
                i: int, 
                population: list[Solution], 
                neighborhoods: np.ndarray, 
                problem: Problem,
                **kwargs) -> Solution:
        
        # 1. Selección de Padres Híbrida
        if np.random.rand() < self.mating_prob:
            # Explotación (Vecindario)
            p_indices = neighborhoods[i]
            parent_indices = np.random.choice(p_indices, 2, replace=False)
        else:
            # Exploración (Global)
            n_pop = len(population)
            parent_indices = np.random.choice(n_pop, 2, replace=False)
            
        parent1 = population[parent_indices[0]]
        parent2 = population[parent_indices[1]]
        
        # 2. Generar Hijo
        child = self.crossover_op.execute(parent1, parent2, problem.bounds)
        child = self.mutation_op.execute(child, problem.bounds)
        
        return child
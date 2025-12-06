import numpy as np

from ..crossovers import Crossover
from .EvolutionaryOperator import EvolutionaryOperator
from ..mutations.Mutation import Mutation
from ..problems import Problem
from ..solutions.Solution import Solution

class StandardOperators(EvolutionaryOperator):
    """
    Implementa la reproducción usando SBX + Mutación Polinómica
    y selección de padres híbrida.
    """
    def __init__(self, 
                 crossover: Crossover, 
                 mutation: Mutation):
        self.crossover_op = crossover
        self.mutation_op = mutation

    def execute(self, 
                i: int, 
                population: list[Solution], 
                neighborhoods: np.ndarray, 
                problem: Problem) -> Solution:
            
        # Asegurarse de que p_indices tenga al menos 2 elementos
        if len(p_indices) < 2:
             p_indices = np.arange(len(population))
             
        parent_indices = np.random.choice(p_indices, 2, replace=False)
        parent1 = population[parent_indices[0]]
        parent2 = population[parent_indices[1]]
        
        # 2. Generar Hijo
        child = self.crossover_op.execute(parent1, parent2, problem.bounds)
        child = self.mutation_op.execute(child, problem.bounds)
        
        return child
import numpy as np

from evolutionary_operator import EvolutionaryOperator
from problems.Problem import Problem
from solutions import Solution

class DifferentialEvolution(EvolutionaryOperator):
    """
    Implementa la reproducción usando Evolución Diferencial (MOEA/D-DE).
    """
    def __init__(self, F: float = 0.5, CR: float = 1.0):
        self.F = F
        self.CR = CR

    def execute(self, 
                i: int, 
                population: list[Solution], 
                neighborhoods: np.ndarray, 
                problem: Problem,
                **kwargs) -> Solution:
        
        # 1. Lógica DE (DE/rand/1/bin)
        
        # a) Seleccionar 3 padres del VECINDARIO
        p_indices = neighborhoods[i]
        r1, r2, r3 = np.random.choice(p_indices, 3, replace=False)
        
        x_r1 = population[r1].variables
        x_r2 = population[r2].variables
        x_r3 = population[r3].variables
        x_i = population[i].variables # Solución "target" actual
        
        # b) Vector Mutante
        bounds = problem.bounds
        min_b = np.array([b[0] for b in bounds])
        max_b = np.array([b[1] for b in bounds])
        
        v = x_r1 + self.F * (x_r2 - x_r3)
        v = np.clip(v, min_b, max_b)

        # c) Cruce Binomial
        n_vars = len(x_i)
        j_rand = np.random.randint(0, n_vars)
        rand_matrix = np.random.rand(n_vars)
        
        child_vars = np.where(rand_matrix < self.CR, v, x_i)
        child_vars[j_rand] = v[j_rand]
        
        # 2. Crear el objeto Solución hijo
        child = problem.create_solution() # Crea un contenedor
        child.variables = child_vars
        
        return child
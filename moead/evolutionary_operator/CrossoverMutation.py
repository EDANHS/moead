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
                debugger=None,
                **kwargs) -> Solution:
        
        debug_context = {
            'index': i,
            'mating_prob': self.mating_prob,
            'population_size': len(population)
        }
        if debugger is not None:
            debugger.start_step('offspring_generation', debug_context)

        # 1. SELECCIÓN DE PADRES HÍBRIDA CON SALVAGUARDAS DE MUESTREO
        if np.random.rand() < self.mating_prob:
            # --- FLUJO DE EXPLOTACIÓN (Vecindario) ---
            p_indices = neighborhoods[i]
            
            # CONTROL DE BORDE: Vecindario demasiado pequeño para muestreo sin reemplazo
            if len(p_indices) < 2:
                p_indices = np.arange(len(population))
            
            # Si incluso la población global es insuficiente (caso pop=1 en pruebas)
            if len(p_indices) < 2:
                # Muestreo con reemplazo forzado: el único individuo disponible se cruzará consigo mismo
                parent_indices = np.random.choice(p_indices, 2, replace=True)
            else:
                parent_indices = np.random.choice(p_indices, 2, replace=False)
        else:
            # --- FLUJO DE EXPLORACIÓN (Global) ---
            n_pop = len(population)
            
            # CONTROL DE BORDE: Población global insuficiente para muestreo sin reemplazo
            if n_pop < 2:
                parent_indices = np.random.choice(n_pop, 2, replace=True)
            else:
                parent_indices = np.random.choice(n_pop, 2, replace=False)
                
        parent1 = population[parent_indices[0]]
        parent2 = population[parent_indices[1]]
        
        if debugger is not None:
            debugger.record_event('offspring_generation', 'info', 'Selected parents', {
                'parent_indices': parent_indices.tolist(),
            })
        
        # 2. Generar Hijo (Continúa el flujo normal sin alteraciones)
        child = self.crossover_op.execute(
            parent1, parent2, problem.bounds,
            debugger=debugger if debugger is not None else None
        )
        child = self.mutation_op.execute(child, problem.bounds)

        # Validación post-mutation
        vars_arr = np.asarray(child.variables)
        min_b = np.asarray([b[0] for b in problem.bounds])
        max_b = np.asarray([b[1] for b in problem.bounds])
        if np.any(vars_arr < min_b) or np.any(vars_arr > max_b):
            child.objectives = np.full(child.objectives.shape, np.inf)
            if child.constraints.shape[0] > 0:
                child.constraints = np.full(child.constraints.shape, np.inf)
            setattr(child, '_invalid_genotype', True)
            if debugger is not None:
                debugger.warning_step('offspring_generation', 'Hijo inválido después de mutación', {
                    'variables': vars_arr.tolist()
                })
        elif debugger is not None:
            debugger.pass_step('offspring_generation', 'Offspring generation completed successfully', {
                'child_variables': vars_arr.tolist()
            })

        return child
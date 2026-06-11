import numpy as np

from ..evolutionary_operator import EvolutionaryOperator
from ..problems.Problem import Problem
from ..solutions import Solution

class DifferentialEvolution(EvolutionaryOperator):
    """
    Implementa la reproducción usando Evolución Diferencial (MOEA/D-DE)
    con SELECCIÓN HÍBRIDA para una exploración robusta (ZDT4).
    """
    def __init__(self, 
                 F: float = 0.5, 
                 CR: float = 0.9,
                 selection_prob: float = 0.9): # <-- ¡NUEVO!
        """
        F: Factor de Diferencia (0 a 2)
        CR: Tasa de Cruce (0 a 1)
        selection_prob: Probabilidad de seleccionar padres del
                        vecindario (explotación) vs. la población
                        global (exploración). 0.9 = 90% vecindario.
        """
        self.F = F
        self.CR = CR
        self.selection_prob = selection_prob # <-- ¡NUEVO!

    def execute(self, 
                i: int, 
                population: list[Solution], 
                neighborhoods: np.ndarray, 
                problem: Problem) -> Solution:
        
        # --- 1. SELECCIÓN DE PADRES HÍBRIDA (¡NUEVO!) ---
        
        if np.random.rand() < self.selection_prob:
            # 90% de las veces: EXPLOTACIÓN (Vecindario)
            source_indices = neighborhoods[i]
        else:
            # 10% de las veces: EXPLORACIÓN (Global)
            source_indices = np.arange(len(population))
        
        # Asegurarse de que la fuente tenga al menos 3 padres
        if len(source_indices) < 3:
            source_indices = np.arange(len(population))
            
        # a) Seleccionar 3 padres de la FUENTE (vecindario o global)
        r1, r2, r3 = np.random.choice(source_indices, 3, replace=False)
        
        x_r1 = population[r1].variables
        x_r2 = population[r2].variables
        x_r3 = population[r3].variables
        x_i = population[i].variables # Solución "target" actual
        
        # --- 2. Lógica DE (Sin cambios) ---
        
        bounds = problem.bounds
        min_b = np.array([b[0] for b in bounds])
        max_b = np.array([b[1] for b in bounds])
        
        # a) Calcular la diferencia pura entre los donantes
        diff = x_r2 - x_r3
        
        # b) Aplicar el factor F y redondear al entero más cercano
        scaled_diff = self.F * diff
        mut_step = np.round(scaled_diff)
        
        # c) REFUERZO DE INERCIA MÍNIMA: 
        # Si había diferencia entre los padres, pero el factor F la redujo a 0,
        # forzamos un salto mínimo de magnitud 1 en la dirección original.
        mut_step = np.where((diff != 0) & (mut_step == 0), np.sign(diff), mut_step)

        # d) Generar vector mutante discreto y reparar límites
        v = x_r1 + mut_step
        v = np.clip(v, min_b, max_b)

        # c) Cruce Binomial
        n_vars = len(x_i)
        j_rand = np.random.randint(0, n_vars)
        rand_matrix = np.random.rand(n_vars)
        
        child_vars = np.where(rand_matrix < self.CR, v, x_i)
        child_vars[j_rand] = v[j_rand]
        
        # 3. Crear el objeto Solución hijo
        child = problem.create_solution()
        child.variables = child_vars
        
        return child
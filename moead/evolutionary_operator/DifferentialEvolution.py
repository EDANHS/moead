import numpy as np
import traceback

from ..evolutionary_operator import EvolutionaryOperator
from ..problems.Problem import Problem
from ..solutions import Solution

class DifferentialEvolution(EvolutionaryOperator):
    """
    Implementa la reproducción usando Evolución Diferencial (MOEA/D-DE)
    con SELECCIÓN HÍBRIDA para una exploración robusta en el hiperespacio NAS.
    Incluye telemetría de frontera y contención de fallos (Fault Tolerance).
    """
    def __init__(self, 
                 F: float = 0.5, 
                 CR: float = 0.9,
                 selection_prob: float = 0.9):
        """
        F: Factor de Diferencia / Perturbación (0 a 2)
        CR: Tasa de Cruce Binomial (0 a 1)
        selection_prob: Probabilidad de seleccionar padres del vecindario (explotación).
        """
        self.F = F
        self.CR = CR
        self.selection_prob = selection_prob

    def execute(self, 
                i: int, 
                population: list[Solution], 
                neighborhoods: np.ndarray, 
                problem: Problem,
                debugger=None,
                **kwargs) -> Solution:
        
        # Inicio del cronómetro transaccional
        if debugger is not None:
            debugger.start_step('offspring_generation_de', {'index': i, 'CR': self.CR, 'F': self.F})
            
        try:
            # --- 1. SELECCIÓN DE PADRES HÍBRIDA ---
            if np.random.rand() < self.selection_prob:
                # 90% de las veces: EXPLOTACIÓN (Vecindario)
                source_indices = neighborhoods[i]
            else:
                # 10% de las veces: EXPLORACIÓN (Global)
                source_indices = np.arange(len(population))
            
            # Asegurarse de que la fuente tenga al menos 3 padres viables
            if len(source_indices) < 3:
                source_indices = np.arange(len(population))
                
            # Seleccionar 3 padres de la FUENTE (vecindario o global)
            r1, r2, r3 = np.random.choice(source_indices, 3, replace=False)
            
            x_r1 = population[r1].variables
            x_r2 = population[r2].variables
            x_r3 = population[r3].variables
            x_i = population[i].variables # Solución "target" actual
            
            # --- 2. LÓGICA DE EVOLUCIÓN DIFERENCIAL ---
            bounds = problem.bounds
            min_b = np.array([b[0] for b in bounds])
            max_b = np.array([b[1] for b in bounds])
            
            # a) Calcular la diferencia pura direccional entre los donantes
            diff = x_r2 - x_r3
            
            # b) Aplicar el factor de perturbación (F) y redondear al dominio discreto
            scaled_diff = self.F * diff
            mut_step = np.round(scaled_diff)
            
            # c) REFUERZO DE INERCIA MÍNIMA
            mut_step = np.where((diff != 0) & (mut_step == 0), np.sign(diff), mut_step)

            # d) Generar vector mutante (Raw)
            v_raw = x_r1 + mut_step
            
            # e) Reparación Geométrica (Bounds Clipping)
            v_clipped = np.clip(v_raw, min_b, max_b)

            # [TELEMETRÍA DE FRONTERA] Auditoría de impacto de bordes
            clipping_occurred = not np.array_equal(v_raw, v_clipped)
            if clipping_occurred and debugger is not None:
                debugger.warning_step('offspring_generation_de', 
                                      'Reparación geométrica aplicada (Boundary Clipping).', 
                                      context={'v_raw': v_raw.tolist(), 'v_clipped': v_clipped.tolist()})

            # f) Cruce Binomial (Preservación del fenotipo dominante)
            n_vars = len(x_i)
            j_rand = np.random.randint(0, n_vars)
            rand_matrix = np.random.rand(n_vars)
            
            child_vars = np.where(rand_matrix < self.CR, v_clipped, x_i)
            child_vars[j_rand] = v_clipped[j_rand] # Garantiza al menos una mutación pura
            
            # 3. Creación y validación del objeto Solución hijo
            child = problem.create_solution()
            child.variables = child_vars
            
            # Cierre exitoso del cronómetro
            if debugger is not None:
                debugger.pass_step('offspring_generation_de', 'Vector Mutante DE generado con éxito.')
                
            return child

        except Exception as e:
            # [CONTENCIÓN DE FALLOS] Captura y registro de caídas críticas sin interrumpir el orquestador
            if debugger is not None:
                error_trace = traceback.format_exc()
                debugger.fail_step('offspring_generation_de', e, 
                                   'Fallo crítico durante la generación del vector DE.', 
                                   context={'traceback': error_trace})
            
            # Fallback de seguridad: Instanciación limpia y delegación de propiedades
            # Esto evita que Keras evalúe basura y garantiza la continuidad del experimento
            child_fallback = problem.create_solution()
            
            # El setter de DLSolution (@variables.setter) ya aplica aislamiento de memoria (np.copy)
            child_fallback.variables = population[i].variables
            
            # Respeto estricto del Contrato de Interfaz mediante la propiedad pública
            if hasattr(child_fallback, 'invalid_genotype'):
                child_fallback.invalid_genotype = True
            else:
                setattr(child_fallback, '_invalid_genotype', True)
            
            return child_fallback
import numpy as np
from moead.solutions import Solution
from ..evolutionary_operator import EvolutionaryOperator
from ..problems.Problem import Problem
# Asumimos la importación de tu surrogate
# from ZCPSurrogate import SurrogatePredictor

class ZCPMoveProposal(EvolutionaryOperator):
    """
    Decorador Estratégico para Operadores Evolutivos (Move Proposal).
    Envuelve cualquier operador base (DE, CrossoverMutation, etc.) para generar 
    múltiples descendientes, filtrándolos instantáneamente mediante un modelo 
    subrogado y retornando únicamente al candidato con mayor aptitud teórica.
    """
    def __init__(self, 
                 base_operator: EvolutionaryOperator, 
                 surrogate_model, 
                 pool_size: int = 10):
        """
        :param base_operator: La instancia del operador original (ej. DifferentialEvolution).
        :param surrogate_model: Instancia pre-entrenada de ZCPSurrogate.
        :param pool_size: Cantidad de hijos a generar y evaluar por cada cruce.
        """
        self.base_operator = base_operator
        self.surrogate = surrogate_model
        self.pool_size = pool_size

    def execute(self, 
                i: int, 
                population: list[Solution], 
                neighborhoods: np.ndarray, 
                problem: Problem,
                debugger=None,
                **kwargs) -> Solution:
        
        if debugger is not None:
            debugger.start_step('zcp_move_proposal', {
                'pool_size': self.pool_size,
                'base_operator': self.base_operator.__class__.__name__
            })

        best_child = None
        best_predicted_loss = float('inf')
        valid_candidates_generated = 0

        # 1. Muestreo de Descendencia Múltiple
        for _ in range(self.pool_size):
            # Delegamos la matemática estocástica (F, CR, SBX) al operador original
            child = self.base_operator.execute(
                i=i, population=population, neighborhoods=neighborhoods, 
                problem=problem, debugger=None, **kwargs
            )

            # Descartamos inmediatamente mutaciones que rompieron la barrera geométrica
            if getattr(child, '_invalid_genotype', False):
                continue
                
            valid_candidates_generated += 1

            # 2. Inferencia Subrogada Instantánea
            # Decodificamos el genotipo a hiperparámetros
            config = problem.decode_solution(child.variables)
            
            # El Random Forest predice el Dice Loss en microsegundos
            predicted_loss = self.surrogate.predict_loss(config)

            # 3. Torneo de Supervivencia Interno
            if predicted_loss < best_predicted_loss:
                best_predicted_loss = predicted_loss
                best_child = child

        # --- Control de Fallos (Fault Tolerance) ---
        if best_child is None:
            if debugger is not None:
                debugger.warning_step('zcp_move_proposal', 'Ningún candidato del pool fue geométricamente válido.')
            
            # Retornamos un individuo fallido para que el orquestador principal lo omita
            best_child = problem.create_solution()
            setattr(best_child, '_invalid_genotype', True)
            return best_child

        if debugger is not None:
            debugger.pass_step('zcp_move_proposal', 'Mejor candidato seleccionado del pool.', {
                'valid_candidates': valid_candidates_generated,
                'predicted_best_loss': best_predicted_loss,
                'selected_variables': best_child.variables.tolist()
            })

        # Retornamos al hijo vencedor. Para el MOEA/D, fue como si el operador
        # hubiera sido extremadamente inteligente y afortunado en su primer intento.
        return best_child
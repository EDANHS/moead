import numpy as np
import multiprocessing
from moead.solutions import Solution
from ..evolutionary_operator import EvolutionaryOperator
from ..problems.Problem import Problem
from moead.utils import zcp_evaluation_worker

class ZCPMoveProposal(EvolutionaryOperator):
    """
    Decorador Estratégico para Operadores Evolutivos (Move Proposal).
    
    Metodología Integral:
    Envuelve cualquier operador base (DE, UX, etc.) para generar una cohorte de 
    descendientes (pool). Implementa una extracción telemetrica en tiempo real de 
    los Zero-Cost Proxies para cada descendiente, garantizando que el modelo 
    subrogado opere sobre un vector hiperdimensional completo antes de aplicar
    la presión selectiva (torneo de aptitud teórica).
    """
    def __init__(self, 
                 base_operator: EvolutionaryOperator, 
                 surrogate_model, 
                 pool_size: int = 10):
        """
        :param base_operator: Operador estocástico original (ej. DifferentialEvolution).
        :param surrogate_model: Instancia pre-entrenada de ZCPSurrogate.
        :param pool_size: Amplitud de la exploración (candidatos generados por iteración).
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
        
        # Aislamiento de subprocesos para evaluación dinámica
        ctx = multiprocessing.get_context("spawn")

        # 1. Muestreo de Descendencia Múltiple y Exploración Local
        for _ in range(self.pool_size):
            # Delegamos la perturbación genotípica (F, CR, SBX) al operador base
            child = self.base_operator.execute(
                i=i, population=population, neighborhoods=neighborhoods, 
                problem=problem, debugger=None, **kwargs
            )

            # Saneamiento geométrico inmediato
            if getattr(child, '_invalid_genotype', False):
                continue
                
            config = problem.decode_solution(child.variables)
            
            # 2. Extracción Telemetrica Estructural (Real-Time ZCP)
            # Evaluar la topología efímera sin instanciar entrenamientos pesados
            queue = ctx.Queue()
            use_gpu_flag = getattr(problem, 'use_gpu', True)
            
            process = ctx.Process(
                target=zcp_evaluation_worker,
                args=(queue, config, problem.input_shape, problem.max_trainable_params, use_gpu_flag, "ensemble")
            )
            process.start()
            
            try:
                # Tolerancia de latencia ajustada para operativas intraloop
                result = queue.get(timeout=30) 
                
                # Integración de la huella estructural en la configuración
                if result.get("success", False):
                    config['zcp_synflow'] = result.get('zcp_synflow', 0.0)
                    config['zcp_snip'] = result.get('zcp_snip', 0.0)
                    config['zcp_jacobian'] = result.get('zcp_jacobian', 0.0)
                else:
                    config['zcp_synflow'] = 0.0
                    config['zcp_snip'] = 0.0
                    config['zcp_jacobian'] = 0.0
            except Exception:
                config['zcp_synflow'] = 0.0
                config['zcp_snip'] = 0.0
                config['zcp_jacobian'] = 0.0
            finally:
                process.join()

            valid_candidates_generated += 1

            # 3. Inferencia Subrogada Plenamente Informada
            predicted_loss = self.surrogate.predict_loss(config)

            # 4. Presión Selectiva (Torneo de Supervivencia Interno)
            if predicted_loss < best_predicted_loss:
                best_predicted_loss = predicted_loss
                best_child = child
                
                # 5. Persistencia de Estado Propagado
                # Inyectamos el diccionario ZCP en la ontología del objeto Solution
                # Esto previene recalculos redundantes cuando el orquestador principal 
                # invoque a problem.evaluate(child)
                setattr(best_child, 'zcp_dict', {
                    'zcp_synflow': config['zcp_synflow'],
                    'zcp_snip': config['zcp_snip'],
                    'zcp_jacobian': config['zcp_jacobian']
                })

        # --- Mitigación de Fallos (Fault Tolerance) ---
        if best_child is None:
            if debugger is not None:
                debugger.warning_step('zcp_move_proposal', 'Anomalía estructural absoluta en el pool de candidatos.')
            
            best_child = problem.create_solution()
            setattr(best_child, '_invalid_genotype', True)
            return best_child

        if debugger is not None:
            debugger.pass_step('zcp_move_proposal', 'Élite seleccionada con precisión subrogada integral.', {
                'valid_candidates': valid_candidates_generated,
                'predicted_best_loss': best_predicted_loss
            })

        return best_child
import sys
import time

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

    def _sanitize_metric(self, value) -> float:
        """
        Escudo protector contra anomalías topológicas. 
        Si el cálculo de gradientes colapsa (NaN/Inf), se penaliza la 
        arquitectura con 0.0 estructural, desplazándola al fondo del ranking.
        """
        if value is None or np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    
    def _force_print(self, message: str):
        """Utilidad de log forzado para evadir el buffering del sistema OS."""
        sys.stdout.write(message + '\n')
        sys.stdout.flush()
    
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
            start_eval = time.perf_counter()
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
                
                synflow = self._sanitize_metric(result.get('zcp_synflow', 0.0))
                snip = self._sanitize_metric(result.get('zcp_snip', 0.0))
                jacobian = self._sanitize_metric(result.get('zcp_jacobian', 0.0))
                # Integración de la huella estructural en la configuración
                if result.get("success", False):
                    config['zcp_synflow'] = synflow
                    config['zcp_snip'] = snip
                    config['zcp_jacobian'] = jacobian
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
            
            raw_params = problem._calculate_params_analytical(config)
                    
            # Evitamos posibles divisiones por cero con variables no inicializadas
            z_min = problem.z_min_params if hasattr(problem, 'z_min_params') else 53
            z_max = problem.z_max_params if hasattr(problem, 'z_max_params') else  35000000.0
            obj_params_norm = float(np.clip((raw_params - z_min) / (z_max - z_min), 0.0, 1.0))

            elapsed_time = time.perf_counter() - start_eval

            self._force_print(f"\n--> [CACHE MISS] Evaluando arquitectura: {config}")
            self._force_print(f"    Resultados -> Dice Loss Pred: {predicted_loss:.4f} | Params Norm: {obj_params_norm:.4f} | Latencia: {elapsed_time:.4f}s")
            self._force_print(f"    [OK] Arq {_:04d} | ZCP-Synflow: {synflow:.2e} | ZCP-SNIP: {snip:.2e} | ZCP-Jacobian: {jacobian:.2e}")
            
            
            # 4. Presión Selectiva (Torneo de Supervivencia Interno)
            setattr(child, 'zcp_metrics', {
                'zcp_synflow': config['zcp_synflow'],
                'zcp_snip': config['zcp_snip'],
                'zcp_jacobian': config['zcp_jacobian']
            })

            if predicted_loss < best_predicted_loss:
                best_predicted_loss = predicted_loss
                best_child = child

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
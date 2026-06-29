import time
import sys
import numpy as np
import multiprocessing
from moead.utils import zcp_evaluation_worker

class ZeroCostWarmup:
    """
    Encapsula la estrategia de inicialización inteligente (Zero-Cost Warmup).
    
    Metodología:
    Implementa un criterio de selección de cuatro ejes jerárquicos:
    1. Synflow (Expresividad) - Máximo
    2. SNIP (Entrenabilidad) - Máximo
    3. Jacobiano (Diversidad Espacial) - Máximo
    4. Modelo Subrogado (Dice Loss Predicho) - Mínimo (Desempate)
    
    Esta estrategia asegura que la población inicial sea robusta, entrenable y
    optimizada hacia el rendimiento esperado. Incluye mecanismos de saneamiento 
    defensivo para evitar que arquitecturas inestables (NaNs) corrompan el 
    ordenamiento darwiniano.
    """
    def __init__(self, warmup_size: int = 1000, verbose: int = 1):
        self.warmup_size = warmup_size
        self.verbose = verbose

    def _force_print(self, message: str):
        """Utilidad de log forzado para evadir el buffering del sistema OS."""
        sys.stdout.write(message + '\n')
        sys.stdout.flush()

    def _sanitize_metric(self, value) -> float:
        """
        Escudo protector contra anomalías topológicas. 
        Si el cálculo de gradientes colapsa (NaN/Inf), se penaliza la 
        arquitectura con 0.0 estructural, desplazándola al fondo del ranking.
        """
        if value is None or np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)

    def execute(self, problem, n_pop: int, history=None) -> list:
        self._force_print(f"\n[WARMUP] Iniciando prospección estructural multicriterio. Muestreando {self.warmup_size} topologías...")
        start_time = time.time()
        
        candidate_pool = []
        
        # 1. Generación estocástica (Fase de Exploración)
        while len(candidate_pool) < self.warmup_size:
            sol = problem.create_solution()
            if getattr(sol, '_invalid_genotype', False) or not problem._is_within_bounds(sol.variables):
                continue
            config = problem.decode_solution(sol.variables)
            candidate_pool.append({'solution': sol, 'config': config, 'zcp_metrics': None, 'pred_dice': None})

        self._force_print(f"[WARMUP] Orquestando procesos de evaluación con telemetría activada...")

        # 2. Orquestación Paralela Aislada
        ctx = multiprocessing.get_context("spawn")
        for idx, candidate in enumerate(candidate_pool):
            if self.verbose >= 1 and idx % 50 == 0:
                self._force_print(f"--> [PROSPECCIÓN] Progreso: {idx}/{self.warmup_size} arquitecturas procesadas.")
                
            start_eval = time.perf_counter()
            queue = ctx.Queue()
            
            process = ctx.Process(
                target=zcp_evaluation_worker,
                args=(queue, candidate['config'], problem.input_shape, problem.max_trainable_params, problem.use_gpu, "ensemble")
            )
            process.start()
            
            try:
                result = queue.get(timeout=60)
                elapsed_time = time.perf_counter() - start_eval
                
                if result.get("success", False):
                    # 1. Extracción y Saneamiento Defensivo
                    synflow = self._sanitize_metric(result.get('zcp_synflow', 0.0))
                    snip = self._sanitize_metric(result.get('zcp_snip', 0.0))
                    jacobian = self._sanitize_metric(result.get('zcp_jacobian', 0.0))

                    # 2. Vectorización Jerárquica (Negados para maximizar en el sort ascendente)
                    candidate['zcp_metrics'] = (-synflow, -snip, -jacobian)
                    
                    # 3. Telemetría de Oráculo y Cálculo Analítico
                    config = candidate['config']
                    candidate['pred_dice'] = problem.surrogate.predict_loss(config)
                    raw_params = problem._calculate_params_analytical(config)
                    
                    # Evitamos posibles divisiones por cero con variables no inicializadas
                    z_min = problem.z_min_params if hasattr(problem, 'z_min_params') else 53
                    z_max = problem.z_max_params if hasattr(problem, 'z_max_params') else  35000000.0
                    obj_params_norm = float(np.clip((raw_params - z_min) / (z_max - z_min), 0.0, 1.0))
            
                    if self.verbose >= 1:
                        self._force_print(f"\n--> [CACHE MISS] Evaluando arquitectura: {config}")
                        self._force_print(f"    Resultados -> Dice Loss Pred: {candidate['pred_dice']:.4f} | Params Norm: {obj_params_norm:.4f} | Latencia: {elapsed_time:.4f}s")
                        self._force_print(f"    [OK] Arq {idx:04d} | ZCP-Synflow: {synflow:.2e} | ZCP-SNIP: {snip:.2e} | ZCP-Jacobian: {jacobian:.2e}")
                else:
                    if self.verbose >= 1:
                        self._force_print(f"    [!] Arq {idx:04d} Fallida (Penalizando): {result.get('error', 'Desconocido')}")
            
            except Exception as e:
                if self.verbose >= 1:
                    self._force_print(f"    [ERROR] Worker {idx:04d} no respondió: {e}")
            finally:
                process.join()

        # 3. Selección Darwiniana Jerárquica (Red de Seguridad)
        valid_pool = [c for c in candidate_pool if c['zcp_metrics'] is not None and c['pred_dice'] is not None]
        
        if not valid_pool:
            self._force_print("[CRÍTICO] Warmup: Ninguna arquitectura superó los filtros de robustez. Rescatando población cruda...")
            elite_solutions = [c['solution'] for c in candidate_pool[:n_pop]]
        else:
            # El sort utiliza la tupla combinada. Como ZCP está negado, actúa como maximizador 
            # para la topología y como minimizador para el Dice Loss.
            valid_pool.sort(key=lambda x: (x['zcp_metrics'], x['pred_dice']))
            elite_solutions = [c['solution'] for c in valid_pool[:n_pop]]
            
        self._force_print(f"\n[WARMUP] Selección jerárquica completada en {time.time() - start_time:.2f}s. {len(elite_solutions)} individuos de élite seleccionados.")
        return elite_solutions
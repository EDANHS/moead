import time
import numpy as np
import multiprocessing
from moead.utils import zcp_evaluation_worker

class ZeroCostWarmup:
    """
    Encapsula la estrategia de inicialización inteligente (Zero-Cost Warmup).
    Implementa un criterio de selección de cuatro ejes jerárquicos:
    1. Synflow (Expresividad) - Máximo
    2. SNIP (Entrenabilidad) - Máximo
    3. Jacobiano (Diversidad Espacial) - Máximo
    4. Modelo Subrogado (Dice Loss Predicho) - Mínimo (Desempate)
    
    Esta estrategia asegura que la población inicial sea robusta, entrenable y
    optimizada hacia el rendimiento esperado, garantizando una convergencia más rápida.
    """
    def __init__(self, warmup_size: int = 1000, verbose: int = 1):
        self.warmup_size = warmup_size
        self.verbose = verbose

    def execute(self, problem, n_pop: int) -> list:
        """
        Ejecuta el filtrado darwiniano utilizando un ordenamiento jerárquico 
        sobre [Synflow, SNIP, Jacobiano, Pred_Dice], integrando telemetría
        granular para auditoría de rendimiento estructural.
        """
        print(f"\n[WARMUP] Iniciando prospección estructural multicriterio. Muestreando {self.warmup_size} topologías...")
        start_time = time.time()
        
        candidate_pool = []
        
        # 1. Generación estocástica
        while len(candidate_pool) < self.warmup_size:
            sol = problem.create_solution()
            if getattr(sol, '_invalid_genotype', False) or not problem._is_within_bounds(sol.variables):
                continue
            config = problem.decode_solution(sol.variables)
            candidate_pool.append({'solution': sol, 'config': config, 'zcp_metrics': None, 'pred_dice': None})

        print(f"[WARMUP] Analizando robustez multidimensional y rendimiento predicho...")

        # 2. Orquestación con Telemetría Multidimensional
        ctx = multiprocessing.get_context("spawn")
        for idx, candidate in enumerate(candidate_pool):
            if self.verbose >= 1 and idx % 50 == 0:
                print(f"--> [PROSPECCIÓN] Progreso: {idx}/{self.warmup_size} topologías evaluadas.")
                
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
                    # Almacenamos el vector de métricas para ordenamiento jerárquico
                    # Negamos las métricas ZCP para que el sort jerárquico las maximice
                    candidate['zcp_metrics'] = (
                        -result.get('zcp_synflow', 0),
                        -result.get('zcp_snip', 0),
                        -result.get('zcp_jacobian', 0)
                    )
                    
                    # Cálculo de métricas para auditoría (Telemetría de Oráculo)
                    config = candidate['config']
                    candidate['pred_dice'] = problem.surrogate.predict_loss(config)
                    raw_params = problem._calculate_params_analytical(config)
                    obj_params_norm = float(np.clip((raw_params - problem.z_min_params) / (problem.z_max_params - problem.z_min_params), 0.0, 1.0))
            
                    print(f"\n--> [CACHE MISS] Evaluando arquitectura: {config}")
                    print(f"    Resultados -> Dice Loss: {candidate['pred_dice']:.4f} | Parámetros: {obj_params_norm} | T: {elapsed_time:.6f}s")
                
                    print(f"    [OK] Arq {idx:04d} | ZCP: {result.get('zcp_synflow', 0):.2e}")
                else:
                    print(f"    [!] Arq {idx:04d} Fallida: {result.get('error', 'Desconocido')}")
            finally:
                process.join()

        # 3. Selección Darwiniana Jerárquica (Cuádruple Eje)
        valid_pool = [c for c in candidate_pool if c['zcp_metrics'] is not None and c['pred_dice'] is not None]
        
        # El sort utiliza la tupla combinada. Como ZCP está negado, el sort default (ascendente) 
        # actúa como un maximizador para ZCP y minimizador para Dice Loss.
        valid_pool.sort(key=lambda x: (x['zcp_metrics'], x['pred_dice']))
        
        elite_solutions = [cand['solution'] for cand in valid_pool[:n_pop]]
        
        elapsed = time.time() - start_time
        print(f"\n[WARMUP] Selección jerárquica completada en {elapsed:.2f}s. {len(elite_solutions)} individuos seleccionados.")
        
        return elite_solutions
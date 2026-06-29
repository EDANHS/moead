# Nuevo Archivo: ZeroCostWarmup.py
import time
import numpy as np
import multiprocessing
from moead.utils import zcp_evaluation_worker

class ZeroCostWarmup:
    """
    Encapsula la estrategia de inicialización inteligente (Zero-Cost Warmup).
    Genera un pool masivo de soluciones aleatorias y las filtra utilizando
    proxies sin entrenamiento para entregar una población semilla de alto rendimiento.
    """
    def __init__(self, warmup_size: int = 1000):
        """
        :param warmup_size: Cantidad de arquitecturas estocásticas a evaluar antes del filtrado.
        """
        self.warmup_size = warmup_size

    def execute_warmup(self, problem, n_pop: int) -> list:
        """
        Ejecuta el sobre-muestreo y filtrado de la población.
        
        :param problem: Instancia del problema (DLProblemRefactor o DLProblemZCP).
        :param n_pop: Tamaño final de la población requerido por MOEA/D.
        :return: Lista con las N soluciones élite aisladas en memoria.
        """
        print(f"\n[WARMUP] Iniciando prospección estructural. Muestreando {self.warmup_size} topologías...")
        start_time = time.time()
        
        candidate_pool = []
        
        # 1. Generación estocástica de genotipos válidos
        while len(candidate_pool) < self.warmup_size:
            sol = problem.create_solution()
            
            # Barrera defensiva geométrica
            if getattr(sol, '_invalid_genotype', False) or not problem._is_within_bounds(sol.variables):
                continue
                
            config = problem.decode_solution(sol.variables)
            candidate_pool.append({'solution': sol, 'config': config, 'zcp_score': -np.inf})

        print(f"[WARMUP] Analizando entrenabilidad ab initio en {len(candidate_pool)} candidatos...")

        # 2. Orquestación multiproceso aislada mediante 'spawn'
        ctx = multiprocessing.get_context("spawn")
        
        for idx, candidate in enumerate(candidate_pool):
            queue = ctx.Queue()
            
            process = ctx.Process(
                target=zcp_evaluation_worker,
                args=(
                    queue,
                    candidate['config'],
                    problem.input_shape,
                    problem.max_trainable_params,
                    problem.use_gpu,
                    "synflow"  # Métrica de flujo sináptico analítico
                )
            )
            process.start()
            
            try:
                # Tiempo límite preventivo por arquitectura
                result = queue.get(timeout=60)
                if result.get("success", False):
                    candidate['zcp_score'] = result.get("zcp_score", -np.inf)
            except Exception as e:
                pass  # Las soluciones inestables retienen score -inf y se descartan en el ordenamiento
            finally:
                process.join()
                
            if idx > 0 and idx % 200 == 0:
                print(f"  --> {idx}/{len(candidate_pool)} topologías procesadas.")

        # 3. Selección darwiniana del frente de élite
        # Ordenamos de mayor a menor capacidad de flujo de gradientes (Synflow Score)
        candidate_pool.sort(key=lambda x: x['zcp_score'], reverse=True)
        
        # Extraemos estrictamente las N mejores soluciones de la población
        elite_solutions = [cand['solution'] for cand in candidate_pool[:n_pop]]
        
        elapsed = time.time() - start_time
        print(f"[WARMUP] Proceso terminado en {elapsed:.2f}s. {n_pop} individuos seleccionados para la Gen 0.\n")
        
        return elite_solutions
import os
import gc
import json
import time
import numpy as np
import multiprocessing
import queue as py_queue
from pathlib import Path
from datetime import datetime

from moead.solutions import DLSolution
from moead.problems import Problem
from moead.utils import evaluation_worker, bounds_worker
from moead.utils.StepDebugger import StepDebugger

class DLProblemRefactor(Problem):
    """
    Problema de Optimización de Hiperparámetros para Deep Learning.
    OPTIMIZADO PARA RTX 5080: Memoria Caché Autónoma, Acumulación y Control de Verbosidad.
    """
    def __init__(self, 
                 X_data, Y_data,
                 input_shape=(256, 256, 1),
                 train_batch_size: int = 4,
                 gradient_accumulation_steps: int = 2,
                 val_batch_size: int = 8,  
                 epochs: int = 20,
                 patience: int = 5,
                 verbose: int = 1,              # 0: Silencio | 1: Info Clave | 2: Debug/Auditoría
                 cache_path: str | Path | None = None,
                 timeout_per_evaluation: int = 1200,
                 use_gpu: bool = True):
        
        self.input_shape = input_shape
        self.verbose = verbose
        self.timeout_per_evaluation = timeout_per_evaluation
        self.use_gpu = use_gpu

        # 1. Gestión de Datos - Ahora son paths
        self.X_data = X_data
        self.Y_data = Y_data

        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Configuración de memoria GPU
        self.gpu_memory_gb = 16.0  # RTX 5080
        
        # --- CONFIGURACIÓN DE LA PERSISTENCIA DE CACHÉ ---
        self.cache_path = Path(cache_path) if cache_path else None
        self.evaluation_cache = {}
        self._load_cache_from_disk()
        
        # 2. Definir los Espacios de Búsqueda
        self.filters_opts = [i for i in range(2, 129, 2)] 
        self.dropouts_opts = [round(i * 0.05, 2) for i in range(11)] 
        self.kernel_opts = [(1,1), (3,3), (5,5), (7,7)] 
        self.act_opts = ['ReLU', 'ELU', 'LeakyReLU', 'GELU', 'Swish']
        self.norm_opts = ['Batch', 'Layer', 'Instance', 'None'] 
        self.pool_opts = ['Max', 'Average']
        self.upsample_opts = ['TransposeConv', 'BilinearUpsample']
        self.bias_opts = [True, False] 
        
        # 3. Definir los Bounds Numéricos para DE
        self._bounds = [
            (1.0, 7.99),                      
            (0.0, len(self.filters_opts) - 0.01), 
            (0.0, len(self.kernel_opts) - 0.01),  
            (0.0, len(self.act_opts) - 0.01),     
            (0.0, len(self.norm_opts) - 0.01),    
            (0.0, len(self.dropouts_opts) - 0.01),
            (0.0, len(self.bias_opts) - 0.01),    
            (0.0, len(self.pool_opts) - 0.01),    
            (0.0, len(self.upsample_opts) - 0.01) 
        ]
        
        self._n_objectives = 2 
        self._n_constraints = 1 

        debug_dir = Path.cwd() / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
        self.debugger = StepDebugger(debug_dir / f'debug_trace_{datetime.now():%Y%m%d_%H%M%S}.json')

        if self.verbose >= 1:
            print("--> Calculando límites teóricos de parámetros (z_min y z_max)...")
            
        self.z_min_params, self.z_max_params = self._calculate_param_bounds()
        self.max_trainable_params = self.z_max_params + 500_000 
        
        if self.verbose >= 1:
            print(f"    Límites calculados -> Min: {self.z_min_params:,}, Max: {self.z_max_params:,}")
            print(f"    Umbral de Pena de Muerte fijado en: {self.max_trainable_params:,} parámetros")

    def _load_cache_from_disk(self):
        """Hidrata la memoria histórica desde el disco."""
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.evaluation_cache = json.load(f)
                if self.verbose >= 1:
                    print(f"\n--> [PERSISTENCIA] Ecosistema hidratado: {len(self.evaluation_cache)} arquitecturas cargadas.")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"\n--> [WARN] Error leyendo caché: {e}. Inicializando limpia.")
                self.evaluation_cache = {}
        else:
            if self.verbose >= 2:
                print("\n--> [PERSISTENCIA] No se detectó caché. Inicializando repositorio vacío.")
            self.evaluation_cache = {}

    def _save_cache_to_disk(self):
        """Escribe la caché usando Escritura Atómica."""
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self.cache_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.evaluation_cache, f, indent=2, ensure_ascii=False)
                os.replace(temp_path, self.cache_path)
            except Exception as e:
                if self.verbose >= 1:
                    print(f"    [WARN PERSISTENCIA] Error al escribir la caché: {e}")

    def _calculate_param_bounds(self):
        """Usa 'spawn' para no contaminar el Main Process con Keras."""
        min_config = {
            'depth': 1, 'initial_filters': self.filters_opts[0], 'kernel_size': (1,1),
            'activation_name': 'ReLU', 'norm_type': 'None', 'dropout_rate': 0.0,
            'use_bias': False, 'pooling_type': 'Max', 'upsample_type': 'BilinearUpsample'
        }
        max_config = {
            'depth': 7, 'initial_filters': self.filters_opts[-1], 'kernel_size': (7,7),
            'activation_name': 'Swish', 'norm_type': 'Batch', 'dropout_rate': 0.5,
            'use_bias': True, 'pooling_type': 'Max', 'upsample_type': 'TransposeConv'
        }

        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=bounds_worker, args=(q, self.input_shape, min_config, max_config))
        p.start()
        
        try:
            # Evita quedarse pegado si TensorFlow crashea inicializando
            p_min, p_max = q.get(timeout=60) 
        except py_queue.Empty:
            p.terminate()
            p_min, p_max = 53.0, 35000000.0
            
        p.join()
        return p_min, p_max

    @property
    def n_objectives(self) -> int: return self._n_objectives
    @property
    def n_constraints(self) -> int: return self._n_constraints
    @property
    def bounds(self) -> list[tuple[float, float]]: return self._bounds

    def create_solution(self):
        vars = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
        return DLSolution(vars, self.n_objectives, self.n_constraints)

    def decode_solution(self, variables: np.ndarray) -> dict:
        return {
            'depth': int(variables[0]),
            'initial_filters': self.filters_opts[int(variables[1])],
            'kernel_size': self.kernel_opts[int(variables[2])],
            'activation_name': self.act_opts[int(variables[3])],
            'norm_type': self.norm_opts[int(variables[4])],
            'dropout_rate': self.dropouts_opts[int(variables[5])],
            'use_bias': self.bias_opts[int(variables[6])],
            'pooling_type': self.pool_opts[int(variables[7])],
            'upsample_type': self.upsample_opts[int(variables[8])]
        }

    def _is_within_bounds(self, variables: np.ndarray) -> bool:
        vars_arr = np.asarray(variables)
        min_b = np.asarray([b[0] for b in self.bounds])
        max_b = np.asarray([b[1] for b in self.bounds])
        return not (np.any(vars_arr < min_b) or np.any(vars_arr > max_b))

    
    def evaluate(self, solution):
        debe_guardar_cache = False
        config_key = None
        epochs_registradas = 0

        try:
            self.debugger.start_step('evaluate_solution', {
                'variables': solution.variables.tolist() if isinstance(solution.variables, np.ndarray) else solution.variables,
            })

            # FASE 1: BARRERA GENOTÍPICA
            if not self._is_within_bounds(solution.variables):
                if self.verbose >= 1: print("    [WARN] Fail-safe: Arquitectura fuera de límites.")
                solution.objectives = np.full(self.n_objectives, np.inf)
                solution.constraints = np.full(self.n_constraints, np.inf)
                solution.invalid_genotype = True
                return

            config = self.decode_solution(solution.variables)
            config_key = json.dumps(config, sort_keys=True)
            
            # --- CACHE HIT ---
            if config_key in self.evaluation_cache:
                if self.verbose >= 1: print(f"\n--> [CACHE HIT] Arquitectura recuperada: {config}")
                historial = self.evaluation_cache[config_key]
                solution.objectives = np.array(historial['objectives'], dtype=float)
                solution.set_metadata(config=config, training_time=0.0, epoch=int(historial['epoch']))
                
                saved_constraints = historial.get('constraints', [float('inf')] * self.n_constraints)
                solution.constraints = np.array(saved_constraints, dtype=float)
                return

            # --- CACHE MISS ---
            if self.verbose >= 1: print(f"\n--> [CACHE MISS] Evaluando arquitectura: {config}")

            # FASE 2: ORQUESTACIÓN MULTIPROCESSING
            # Uso estricto de 'spawn' para compatibilidad en Linux/TensorFlow
            ctx = multiprocessing.get_context("spawn")
            queue = ctx.Queue()

            worker_args = (
                queue, config,
                self.X_data, self.Y_data,
                self.input_shape,
                self.train_batch_size,
                self.gradient_accumulation_steps,
                self.val_batch_size,
                self.epochs,
                self.patience,
                self.max_trainable_params,
                self.verbose,
                self.use_gpu
            )

            process = ctx.Process(target=evaluation_worker, args=worker_args)
            process.start()
            
            # Bloqueo síncrono hasta que el worker inyecte el resultado
            result = queue.get(timeout=self.timeout_per_evaluation)  # Timeout para evitar bloqueos indefinidos
            process.join()

            # FASE 3: MAPEO DE RESULTADOS
            if not result["success"]:
                error_msg = result["error"]
                if error_msg == "OOM_PREVENTION":
                    if self.verbose >= 1: print(f"    Arquitectura masiva ({result['params']:,}). Evitando OOM y penalizando.")
                elif "OOM" in error_msg:
                    if self.verbose >= 1: print(f"    [OOM] Memoria excedida en GPU. Penalizando arquitectura.")
                else:
                    if self.verbose >= 1: print(f"    [ERROR] El worker falló: {str(error_msg)[:100]} - Penalizando.")
                
                solution.objectives = np.array([1.0, 1.0])
                solution.constraints = np.full(self.n_constraints, np.inf)
                solution.set_metadata(config=config)
                debe_guardar_cache = True
                return

            # Caso de Éxito
            raw_params = result["params"]
            final_val_dice = result["dice"]
            elapsed_time = result["elapsed"]
            epochs_registradas = result["epochs"]

            # --- CÁLCULO DE OBJETIVOS ---
            obj_dice_loss = float(1.0 - final_val_dice)
            
            # NUEVO: ESCUDO ANTI-NaN / EXPLOSIÓN DE GRADIENTES
            if np.isnan(obj_dice_loss) or np.isinf(obj_dice_loss):
                if self.verbose >= 1: 
                    print("    [WARN] Arquitectura inestable (Gradients Exploded -> NaN). Penalizando.")
                obj_dice_loss = 1.0  # Peor valor posible

            obj_params_norm = float((raw_params - self.z_min_params) / (self.z_max_params - self.z_min_params))
            
            # NUEVO: ESCUDO ANTI-NaN PARA PARÁMETROS (Por si acaso)
            if np.isnan(obj_params_norm) or np.isinf(obj_params_norm):
                obj_params_norm = 1.0

            obj_params_norm = float(np.clip(obj_params_norm, 0.0, 1.0))

            if self.verbose >= 1:
                print(f"    Resultados -> Dice Loss: {obj_dice_loss:.4f} | Params Norm: {obj_params_norm:.4f} | Tiempo: {elapsed_time:.1f}s")

            solution.objectives = np.array([obj_dice_loss, obj_params_norm])
            solution.constraints = np.zeros(self.n_constraints)
            solution.set_metadata(config=config, training_time=elapsed_time, epoch=epochs_registradas)
            debe_guardar_cache = True

        except multiprocessing.queues.Empty:
            process.terminate()
            process.join()
            result = {"success": False, "error": "HARD_CRASH_OR_TIMEOUT"}

            solution.objectives = np.array([1.0, 1.0])
            solution.constraints = np.full(self.n_constraints, np.inf)
            if 'config' in locals(): solution.set_metadata(config=config)
        except Exception as e:
            if self.verbose >= 1: print(f"    [ERROR ORQUESTADOR] Fallo de comunicación: {str(e)[:100]} - Penalizando.")
            solution.objectives = np.array([1.0, 1.0])
            solution.constraints = np.full(self.n_constraints, np.inf)
            if 'config' in locals(): solution.set_metadata(config=config)
            debe_guardar_cache = True
        
        finally:
            # FASE 4: ESCRITURA TRANSACCIONAL
            if debe_guardar_cache and config_key is not None:
                self.evaluation_cache[config_key] = {
                    'objectives': solution.objectives.tolist(),
                    'constraints': solution.constraints.tolist(),
                    'epoch': epochs_registradas
                }
                self._save_cache_to_disk()

            # Forzar Garbage Collection del lado del orquestador
            gc.collect()
            # Pausa breve para asegurar el flushing de VRAM post-proceso
            time.sleep(2)
import gc
import os
import json
import numpy as np
import tensorflow as tf
import time
from pathlib import Path
from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision

from moead.models import build_unet
from moead.solutions import DLSolution
from moead.problems import Problem
from moead.utils import dice_coefficient, dice_loss
from moead.utils.StepDebugger import StepDebugger

class DLProblem(Problem):
    """
    Problema de Optimización de Hiperparámetros para Deep Learning.
    OPTIMIZADO PARA RTX 5080: Memoria Caché Autónoma, Acumulación y Control de Verbosidad.
    """
    def __init__(self, 
                 X_train, Y_train, 
                 X_val, Y_val, 
                 X_test=None, Y_test=None,
                 input_shape=(256, 256, 1),
                 train_batch_size: int = 4, 
                 val_batch_size: int = 8,  
                 epochs: int = 20,
                 patience: int = 5,
                 verbose: int = 1,              # 0: Silencio | 1: Info Clave | 2: Debug/Auditoría
                 cache_path: str | Path | None = None):
        
        self.input_shape = input_shape
        self.verbose = verbose

        # 1. Gestión de Datos
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Configuración de memoria GPU
        self.gpu_memory_gb = 16.0  # RTX 5080
        mixed_precision.set_global_policy('mixed_float16')
        
        # --- CONFIGURACIÓN DE LA PERSISTENCIA DE CACHÉ ---
        self.cache_path = Path(cache_path) if cache_path else None
        self.evaluation_cache = {}
        self._load_cache_from_disk()
        
        # 2. Definir los Espacios de Búsqueda
        self.filters_opts = [i for i in range(2, 65, 2)] 
        self.dropouts_opts = [round(i * 0.05, 2) for i in range(11)] 
        self.kernel_opts = [(1,1), (3,3), (5,5)] 
        self.act_opts = ['ReLU', 'ELU', 'LeakyReLU', 'GELU', 'Swish']
        self.norm_opts = ['Batch', 'Layer', 'Instance', 'None'] 
        self.pool_opts = ['Max', 'Average']
        self.upsample_opts = ['TransposeConv', 'BilinearUpsample']
        self.bias_opts = [True, False] 
        
        # 3. Definir los Bounds Numéricos para DE
        self._bounds = [
            (1.0, 5.99),                      
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
        min_config = {
            'depth': 1, 'initial_filters': self.filters_opts[0], 'kernel_size': (1,1),
            'activation_name': 'ReLU', 'norm_type': 'None', 'dropout_rate': 0.0,
            'use_bias': False, 'pooling_type': 'Max', 'upsample_type': 'BilinearUpsample'
        }
        max_config = {
            'depth': 5, 'initial_filters': self.filters_opts[-1], 'kernel_size': (5,5),
            'activation_name': 'Swish', 'norm_type': 'Batch', 'dropout_rate': 0.5,
            'use_bias': True, 'pooling_type': 'Max', 'upsample_type': 'TransposeConv'
        }

        try:
            m_min = build_unet(self.input_shape, **min_config)
            p_min = m_min.count_params()
            del m_min
        except: p_min = 200.0

        try:
            m_max = build_unet(self.input_shape, **max_config)
            p_max = m_max.count_params()
            del m_max
        except:
            p_max = 35_000_000.0 
            
        K.clear_session()
        gc.collect()
        return float(p_min), float(p_max)

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

    def _create_dataset(self, x, y, custom_batch_size: int, is_training=True):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            ds = ds.shuffle(buffer_size=min(128, len(x)))
        ds = ds.batch(custom_batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE) 
        return ds
    
    def evaluate(self, solution):
        K.clear_session()
        gc.collect()

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
                
                solution.constraints = np.zeros(self.n_constraints)
                return

            # --- CACHE MISS ---
            if self.verbose >= 1: print(f"\n--> [CACHE MISS] Evaluando arquitectura: {config}")
            
            model = build_unet(self.input_shape, **config)
            raw_params = model.count_params()

            # FASE 2: BARRERA PARAMÉTRICA
            if raw_params > self.max_trainable_params:
                if self.verbose >= 1: print(f"    Arquitectura masiva ({raw_params:,}). Evitando OOM y penalizando.")
                solution.objectives = np.array([1.0, 1.0]) 
                solution.constraints = np.full(self.n_constraints, np.inf)
                solution.set_metadata(config=config)
                debe_guardar_cache = True # Memorizamos el monstruo
                return

            # FASE 3: COMPILACIÓN Y ENTRENAMIENTO
            optimizador_acumulativo = tf.keras.optimizers.Adam(
                learning_rate=0.001,
                gradient_accumulation_steps=self.train_batch_size  
            )

            model.compile(
                optimizer=optimizador_acumulativo,
                loss=dice_loss,
                metrics=[dice_coefficient],
                jit_compile=False
            )

            train_ds = self._create_dataset(self.X_train, self.Y_train, custom_batch_size=self.train_batch_size, is_training=True)
            val_ds = self._create_dataset(self.X_val, self.Y_val, custom_batch_size=self.val_batch_size, is_training=False)

            callbacks = []
            if self.patience > 0:
                stopper = EarlyStopping(monitor='val_loss', patience=self.patience, mode='min', restore_best_weights=True)
                callbacks.append(stopper)

            # Silenciamos el I/O de Keras si no estamos en Debug Mode (verbose=2)
            keras_fit_verbose = 2 if self.verbose >= 2 else 0

            start_time = time.time()
            history = model.fit(
                train_ds,
                validation_data=val_ds, 
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=keras_fit_verbose
            )
            elapsed_time = time.time() - start_time
            epochs_registradas = len(history.history['loss'])

            # FASE 4: METODOLOGÍA DE EVALUACIÓN
            if self.X_test is not None and self.Y_test is not None:
                if self.verbose >= 2: print("    [METODOLOGÍA] Evaluando sobre el conjunto de Prueba aislado.")
                eval_ds = self._create_dataset(self.X_test, self.Y_test, custom_batch_size=self.val_batch_size, is_training=False)
            else:
                if self.verbose >= 2: print("    [METODOLOGÍA] Fallback: Evaluando sobre el conjunto de Validación (X_val).")
                eval_ds = val_ds

            keras_eval_verbose = 1 if self.verbose >= 2 else 0
            eval_results = model.evaluate(eval_ds, verbose=keras_eval_verbose)
            
            final_val_dice = float(eval_results[1])
            obj_dice_loss = float(1.0 - final_val_dice)

            obj_params_norm = float((raw_params - self.z_min_params) / (self.z_max_params - self.z_min_params))
            obj_params_norm = float(np.clip(obj_params_norm, 0.0, 1.0))

            if self.verbose >= 1:
                print(f"    Resultados -> Dice Loss: {obj_dice_loss:.4f} | Params Norm: {obj_params_norm:.4f} | Tiempo: {elapsed_time:.1f}s")

            solution.objectives = np.array([obj_dice_loss, float(obj_params_norm)])

            solution.constraints = np.zeros(self.n_constraints)

            solution.set_metadata(
                config=config,
                training_time=elapsed_time,
                epoch=epochs_registradas
            )
            debe_guardar_cache = True

        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
            if self.verbose >= 1: print(f"    [OOM] Memoria excedida. Penalizando arquitectura.")
            solution.objectives = np.array([1.0, 1.0])
            solution.constraints = np.full(self.n_constraints, np.inf)
            if 'config' in locals(): solution.set_metadata(config=config)
            debe_guardar_cache = True
        
        except Exception as e:
            if self.verbose >= 1: print(f"    [ERROR] Hilo de evaluación falló: {str(e)[:100]} - Penalizando.")
            solution.objectives = np.array([1.0, 1.0])
            solution.constraints = np.full(self.n_constraints, np.inf)
            if 'config' in locals(): solution.set_metadata(config=config)
            debe_guardar_cache = True
        
        finally:
            # FASE 5: ESCRITURA TRANSACCIONAL Y PURGA
            if debe_guardar_cache and config_key is not None:
                self.evaluation_cache[config_key] = {
                    'objectives': solution.objectives.tolist(),
                    'constraints': solution.constraints.tolist(),
                    'epoch': epochs_registradas
                }
                self._save_cache_to_disk()

            if 'model' in locals(): del model
            if 'train_ds' in locals(): del train_ds
            if 'eval_ds' in locals() and self.X_test is not None: del eval_ds
            if 'val_ds' in locals(): del val_ds
            
            K.clear_session()
            gc.collect()
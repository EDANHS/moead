import gc
import json
import numpy as np
import tensorflow as tf
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
    OPTIMIZADO PARA RTX 5080: Usa tf.data pipelines con Prefetching.
    """
    def __init__(self, 
                 X_train, Y_train, 
                 X_val, Y_val, 
                 X_test=None, Y_test=None,
                 input_shape=(256, 256, 1),
                 batch_size: int = 8,  # Ajustado para balancear memoria y estabilidad
                 epochs: int = 20,
                 patience: int = 5):
        
        self.input_shape = input_shape

        # 1. Gestión de Datos
        # Mantenemos las referencias a NumPy, pero las convertiremos a Tensores bajo demanda
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Configuración de memoria GPU
        self.gpu_memory_gb = 12.0  # RTX 5080 ~12GB VRAM
        self.memory_threshold = 0.65  # 65% de VRAM máxima permitida para mayor robustez
        
        # Precisión Mixta Global para reducir memoria VRAM
        mixed_precision.set_global_policy('mixed_float16')
        
        # 2. Definir los Espacios de Búsqueda (EXPANDIDO)
        self.filters_opts = [i for i in range(2, 65, 2)]  # De 2 a 128 en pasos de 2
        self.kernel_opts = [(1,1), (3,3), (5,5)]  # Agregado kernel 7x7
        self.act_opts = ['ReLU', 'ELU', 'LeakyReLU', 'GELU', 'Swish']
        self.norm_opts = ['Batch', 'Layer', 'Instance', 'None']  # Expandido a 4
        self.pool_opts = ['Max', 'Average']
        self.upsample_opts = ['TransposeConv', 'BilinearUpsample']
        self.bias_opts = [True, False] 
        
        # 3. Definir los Bounds Numéricos para DE (EXPANDIDO)
        self._bounds = [
            (1.0, 5.99),                      # 0: Depth [1-5]
            (0.0, len(self.filters_opts) - 0.01), # 1: Filters (3 opciones: 16,32,64)
            (0.0, len(self.kernel_opts) - 0.01),  # 2: Kernel (2 tamaños: 3x3, 5x5)
            (0.0, len(self.act_opts) - 0.01),     # 3: Activation (5 opciones)
            (0.0, len(self.norm_opts) - 0.01),    # 4: Norm (4: Batch, Layer, Instance, None)
            (0.0, 0.7),                       # 5: Dropout [0.0 - 0.7]
            (0.0, len(self.bias_opts) - 0.01),    # 6: Bias
            (0.0, len(self.pool_opts) - 0.01),    # 7: Pooling
            (0.0, len(self.upsample_opts) - 0.01) # 8: UpSample
        ]
        
        self._n_objectives = 2 
        self._n_constraints = 1  # Penalización explícita de violaciones de bounds
        self.max_trainable_params = 25_000_000  # Umbral duro para evitar OOM de topologías enormes

        debug_dir = Path.cwd() / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
        self.debugger = StepDebugger(debug_dir / f'debug_trace_{datetime.now():%Y%m%d_%H%M%S}.json')

        print("Calculando límites teóricos de parámetros (z_min y z_max)...")
        self.z_min_params, self.z_max_params = self._calculate_param_bounds()
        print(f"Límites calculados -> Min: {self.z_min_params:,}, Max: {self.z_max_params:,}")

    def _calculate_param_bounds(self):
        """Calcula min/max params para normalización."""
        min_config = {
            'depth': 1, 'initial_filters': 2, 'kernel_size': (1,1),
            'activation_name': 'ReLU', 'norm_type': 'None', 'dropout_rate': 0.0,
            'use_bias': False, 'pooling_type': 'Max', 'upsample_type': 'BilinearUpsample'
        }

        # Configuración máxima teórica
        max_config = {
            'depth': 5, 'initial_filters': 64, 'kernel_size': (5,5),
            'activation_name': 'Swish', 'norm_type': 'Batch', 'dropout_rate': 0.8,
            'use_bias': True, 'pooling_type': 'Max', 'upsample_type': 'TransposeConv'
        }

        # K.clear_session()  # comentado para no destruir el handle CUDA entre iteraciones
        try:
            m_min = build_unet(self.input_shape, **min_config)
            p_min = m_min.count_params()
            del m_min
        except: p_min = 200.0

        # K.clear_session()  # comentado para no destruir el handle CUDA entre iteraciones
        try:
            m_max = build_unet(self.input_shape, **max_config)
            p_max = m_max.count_params()
            del m_max
        except:
            p_max = 30_000_000.0 # Fallback seguro

        # K.clear_session()  # comentado para no destruir el handle CUDA entre iteraciones
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
        config = {
            'depth': int(variables[0]),
            'initial_filters': self.filters_opts[int(variables[1])],
            'kernel_size': self.kernel_opts[int(variables[2])],
            'activation_name': self.act_opts[int(variables[3])],
            'norm_type': self.norm_opts[int(variables[4])],
            'dropout_rate': float(variables[5]),
            'use_bias': self.bias_opts[int(variables[6])],
            'pooling_type': self.pool_opts[int(variables[7])],
            'upsample_type': self.upsample_opts[int(variables[8])]
        }
        return config

    def _is_within_bounds(self, variables: np.ndarray) -> bool:
        vars_arr = np.asarray(variables)
        min_b = np.asarray([b[0] for b in self.bounds])
        max_b = np.asarray([b[1] for b in self.bounds])
        return not (np.any(vars_arr < min_b) or np.any(vars_arr > max_b))

    def _create_dataset(self, x, y, is_training=True):
        """
        Eager loading: materializa arrays en RAM y crea pipeline estándar.
        Evita complejidad de tf.py_function y sus picos de memoria en GPU.
        """
        # Eager conversion to float32 (already done in runners, but ensure it here)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            ds = ds.shuffle(buffer_size=min(128, len(x)))

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)  # Minimiza la memoria en vuelo en lugar de AUTOTUNE
        return ds

    def evaluate(self, solution):
        # Limpieza agresiva ANTES de empezar
        K.clear_session()
        gc.collect()
        tf.keras.backend.clear_session()
        gc.collect()

        try:
            self.debugger.start_step('evaluate_solution', {
                'variables': solution.variables.tolist() if isinstance(solution.variables, np.ndarray) else solution.variables,
                'n_objectives': self.n_objectives,
                'n_constraints': self.n_constraints,
            })

            if not self._is_within_bounds(solution.variables):
                self.debugger.fail_step('bounds_check', 'invalid genotype', message='Genotipo fuera de límites de bounds')
                print("    Arquitectura inválida: genotipo fuera de límites. Penalizando sin instanciar modelo.")
                solution.objectives = np.full(self.n_objectives, np.inf)
                solution.constraints = np.full(self.n_constraints, np.inf)
                solution.model_config = {'invalid_genotype': True}
                self.debugger.pass_step('evaluate_solution', message='Evaluación terminada con genotipo inválido')
                return

            config = self.decode_solution(solution.variables)
            solution.model_config = config
            print(f"\n--> Evaluando arquitectura: {config}")
            self.debugger.start_step('model_build', {'config': config})

            # --- CONSTRUCCIÓN ---
            model = build_unet(self.input_shape, **config)
            raw_params = model.count_params()
            self.debugger.pass_step('model_build', {'raw_params': int(raw_params)})

            if raw_params > self.max_trainable_params:
                self.debugger.fail_step('model_size_guard', f'Arquitectura demasiado grande ({raw_params:,} parámetros)')
                print(f"    Arquitectura demasiado grande ({raw_params:,} parámetros). Penalizando sin entrenar.")
                solution.objectives = np.array([1.0, 1.0])
                solution.constraints = np.full(self.n_constraints, np.inf)
                del model
                self.debugger.pass_step('evaluate_solution', message='Evaluación terminada con arquitectura demasiado grande')
                return

            self.debugger.start_step('model_compile', {'optimizer': 'adam', 'loss': 'dice_loss'})
            # --- COMPILACIÓN ---
            model.compile(
                optimizer='adam', 
                loss=dice_loss, 
                metrics=[dice_coefficient], 
                jit_compile=False
            )
            self.debugger.pass_step('model_compile', 'Modelo compilado con éxito')

            self.debugger.start_step('data_pipeline', {'batch_size': self.batch_size, 'train_size': int(len(self.X_train)), 'val_size': int(len(self.X_val))})
            # --- PREPARACIÓN DE DATOS (PIPELINE EAGER) ---
            train_ds = self._create_dataset(self.X_train, self.Y_train, is_training=True)
            val_ds = self._create_dataset(self.X_val, self.Y_val, is_training=False)
            self.debugger.pass_step('data_pipeline', 'Pipelines de datos creados correctamente')

            self.debugger.start_step('train_start', {'epochs': self.epochs, 'patience': self.patience})
            # --- ENTRENAMIENTO ---
            callbacks = []
            if self.patience > 0:
                stopper = EarlyStopping(monitor='val_loss', patience=self.patience, mode='min', restore_best_weights=True)
                callbacks.append(stopper)

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=1
            )
            self.debugger.pass_step('train_start', 'Entrenamiento completado')

            self.debugger.start_step('objectives_compute', {
                'history_keys': list(history.history.keys())
            })
            # --- CÁLCULO DE OBJETIVOS ---
            val_dice_scores = history.history['val_dice_coefficient']
            best_val_dice = float(max(val_dice_scores))
            obj_dice_loss = float(1.0 - best_val_dice)

            raw_params = model.count_params()
            obj_params_norm = float((raw_params - self.z_min_params) / (self.z_max_params - self.z_min_params))
            obj_params_norm = float(np.clip(obj_params_norm, 0.0, 1.0))

            print(f"    Resultados -> Dice Loss: {obj_dice_loss:.4f} | Params Norm: {obj_params_norm:.4f}")
            self.debugger.pass_step('objectives_compute', 'Objetivos calculados', {
                'dice_loss': obj_dice_loss,
                'params_norm': obj_params_norm,
            })

            solution.objectives = np.array([obj_dice_loss, float(obj_params_norm)])
            self.debugger.pass_step('evaluate_solution', 'Evaluación exitosa')

        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
            self.debugger.fail_step('oom_or_internal_error', e, message='OOM o error interno durante evaluación')
            print(f"    OOM detectado: {str(e)[:100]} - Penalizando.")
            solution.objectives = np.array([1.0, 1.0])
        
        except Exception as e:
            self.debugger.fail_step('general_error', e, message='Error general durante evaluación')
            print(f"    Error general: {str(e)[:100]} - Penalizando.")
            solution.objectives = np.array([1.0, 1.0])
        
        finally:
            # Destrucción total del modelo para liberar VRAM
            if 'model' in locals():
                del model
            if 'train_ds' in locals():
                del train_ds
            if 'val_ds' in locals():
                del val_ds
            
            # Limpieza agresiva DESPUÉS de terminar
            K.clear_session()
            tf.keras.backend.clear_session()
            gc.collect()
            gc.collect()  # Double gc.collect() para asegurar
            self.debugger.pass_step('cleanup', 'Garbage collection completada y sesión Keras liberada')

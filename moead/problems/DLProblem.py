import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from moead.models import build_unet
from moead.solutions import DLSolution
from moead.problems import Problem
from moead.utils import dice_coefficient, dice_loss

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
                 batch_size: int = 32, # Aumentado por defecto para aprovechar GPU
                 epochs: int = 10):
        
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
        
        # 2. Definir los Espacios de Búsqueda
        self.filters_opts = [4, 8, 16, 32, 64] # Agregado 64 ya que tienes hardware de sobra
        self.kernel_opts = [(1,1), (3,3), (5,5)]
        self.act_opts = ['ReLU', 'ELU', 'LeakyReLU', 'GELU', 'Swish']
        self.norm_opts = ['Batch', 'Instance', 'Group', None]
        self.pool_opts = ['Max', 'Average']
        self.upsample_opts = ['TransposeConv', 'BilinearUpsample']
        self.bias_opts = [True, False]
        
        # 3. Definir los Bounds Numéricos para DE
        self._bounds = [
            (1.0, 4.99),                      # 0: Depth
            (0.0, len(self.filters_opts) - 0.01), # 1: Filters
            (0.0, len(self.kernel_opts) - 0.01),  # 2: Kernel
            (0.0, len(self.act_opts) - 0.01),     # 3: Activation
            (0.0, len(self.norm_opts) - 0.01),    # 4: Norm
            (0.0, 0.6),                       # 5: Dropout (Max 0.6 es más sano)
            (0.0, len(self.bias_opts) - 0.01),    # 6: Bias
            (0.0, len(self.pool_opts) - 0.01),    # 7: Pooling
            (0.0, len(self.upsample_opts) - 0.01) # 8: UpSample
        ]
        
        self._n_objectives = 2 
        self._n_constraints = 0

        print("Calculando límites teóricos de parámetros (z_min y z_max)...")
        self.z_min_params, self.z_max_params = self._calculate_param_bounds()
        print(f"Límites calculados -> Min: {self.z_min_params:,}, Max: {self.z_max_params:,}")

    def _calculate_param_bounds(self):
        """Calcula min/max params para normalización."""
        min_config = {
            'depth': 1, 'initial_filters': 4, 'kernel_size': (1,1),
            'activation_name': 'ReLU', 'norm_type': None, 'dropout_rate': 0.0,
            'use_bias': False, 'pooling_type': 'Max', 'upsample_type': 'BilinearUpsample'
        }

        # Configuración máxima teórica
        max_config = {
            'depth': 4, 'initial_filters': 64, 'kernel_size': (5,5),
            'activation_name': 'ReLU', 'norm_type': 'Batch', 'dropout_rate': 0.0,
            'use_bias': True, 'pooling_type': 'Max', 'upsample_type': 'TransposeConv'
        }

        K.clear_session()
        try:
            m_min = build_unet(self.input_shape, **min_config)
            p_min = m_min.count_params()
            del m_min
        except: p_min = 200.0

        K.clear_session()
        try:
            m_max = build_unet(self.input_shape, **max_config)
            p_max = m_max.count_params()
            del m_max
        except:
            p_max = 30_000_000.0 # Fallback seguro

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

    def _create_dataset(self, x, y, is_training=True):
        # Como x e y ya son float32 (desde el script principal), 
        # from_tensor_slices no necesita hacer Cast.
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        
        dataset = dataset.cache()
        if is_training:
            dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def evaluate(self, solution):
        # Limpieza agresiva antes de empezar
        K.clear_session()
        gc.collect()

        config = self.decode_solution(solution.variables)
        solution.model_config = config
        print(f"\n--> Evaluando arquitectura: {config}")

        try:
            # --- CONSTRUCCIÓN ---
            model = build_unet(self.input_shape, **config)
            
            # --- COMPILACIÓN ---
            # JIT compile=True acelera XLA en tarjetas nuevas
            model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient], jit_compile=False)

            # --- PREPARACIÓN DE DATOS (PIPELINE) ---
            train_ds = self._create_dataset(self.X_train, self.Y_train, is_training=True)
            val_ds = self._create_dataset(self.X_val, self.Y_val, is_training=False)

            # --- ENTRENAMIENTO ---
            # Corregido: Monitor 'val_loss' (min) en lugar de 'val_dice' (min)
            stopper = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True)

            history = model.fit(
                train_ds,          # Usamos el pipeline, no numpy directo
                validation_data=val_ds,
                epochs=self.epochs,
                callbacks=[stopper],
                verbose=0          # 0 para mayor velocidad en consola, 1 para debug
            )

            # --- CÁLCULO DE OBJETIVOS ---
            # Usamos el mejor Dice logrado (max)
            val_dice_scores = history.history['val_dice_coefficient']
            best_val_dice = max(val_dice_scores)
            obj_dice_loss = 1.0 - best_val_dice

            raw_params = model.count_params()
            obj_params_norm = (float(raw_params) - self.z_min_params) / (self.z_max_params - self.z_min_params)
            obj_params_norm = np.clip(obj_params_norm, 0.0, 1.0)

            print(f"    Resultados -> Dice Loss: {obj_dice_loss:.4f} | Params Norm: {obj_params_norm:.4f}")

            solution.objectives = np.array([obj_dice_loss, float(obj_params_norm)])

        except Exception as e:
            print(f"    ERROR CRÍTICO: {str(e)}")
            # Penalización extrema
            solution.objectives = np.array([1.0, 1.0])
        
        finally:
            # Destrucción total del modelo para liberar VRAM
            if 'model' in locals():
                del model
            K.clear_session()
            gc.collect()
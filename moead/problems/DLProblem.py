import gc
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
    OPTIMIZADO PARA RTX 5080: Estrategia Híbrida (Acumulación + Filtro Estricto).
    """
    def __init__(self, 
                 X_train, Y_train, 
                 X_val, Y_val, 
                 X_test=None, Y_test=None,
                 input_shape=(256, 256, 1),
                 train_batch_size: int = 4,  # Lote físico reducido para entrenamiento
                 val_batch_size: int = 8,    # Lote de validación puede ser mayor
                 epochs: int = 20,
                 patience: int = 5):
        
        self.input_shape = input_shape

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
        self.gpu_memory_gb = 16.0  # RTX 5080 tiene 16GB VRAM
        
        # Precisión Mixta Global para reducir memoria VRAM y ancho de banda PCIe
        mixed_precision.set_global_policy('mixed_float16')
        
        # 2. Definir los Espacios de Búsqueda (OPTIMIZADO)
        self.filters_opts = [i for i in range(2, 65, 2)] # Máximo 64 filtros iniciales
        # Dropout discretizado de 0.0 a 0.5 en saltos de 0.05
        self.dropouts_opts = [round(i * 0.05, 2) for i in range(11)] 
        self.kernel_opts = [(1,1), (3,3), (5,5)] # Excluimos (5,5) por ineficiencia paramétrica
        self.act_opts = ['ReLU', 'ELU', 'LeakyReLU', 'GELU', 'Swish']
        self.norm_opts = ['Batch', 'Layer', 'Instance', 'None'] 
        self.pool_opts = ['Max', 'Average']
        self.upsample_opts = ['TransposeConv', 'BilinearUpsample']
        self.bias_opts = [True, False] 
        
        # 3. Definir los Bounds Numéricos para DE
        self._bounds = [
            (1.0, 5.99),                      # 0: Depth [1-5] (Bajado para evitar monstruos)
            (0.0, len(self.filters_opts) - 0.01), # 1: Filters 
            (0.0, len(self.kernel_opts) - 0.01),  # 2: Kernel 
            (0.0, len(self.act_opts) - 0.01),     # 3: Activation
            (0.0, len(self.norm_opts) - 0.01),    # 4: Norm
            (0.0, len(self.dropouts_opts) - 0.01),# 5: Dropout
            (0.0, len(self.bias_opts) - 0.01),    # 6: Bias
            (0.0, len(self.pool_opts) - 0.01),    # 7: Pooling
            (0.0, len(self.upsample_opts) - 0.01) # 8: UpSample
        ]
        
        self._n_objectives = 2 
        self._n_constraints = 1 

        debug_dir = Path.cwd() / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
        self.debugger = StepDebugger(debug_dir / f'debug_trace_{datetime.now():%Y%m%d_%H%M%S}.json')

        print("Calculando límites teóricos de parámetros (z_min y z_max)...")
        self.z_min_params, self.z_max_params = self._calculate_param_bounds()
        
        # FILTRO DINÁMICO: El máximo teórico calculado se convierte en nuestro umbral de Pena de Muerte
        self.max_trainable_params = self.z_max_params + 500_000 # Margen de seguridad leve
        
        print(f"Límites calculados -> Min: {self.z_min_params:,}, Max: {self.z_max_params:,}")
        print(f"Umbral de Pena de Muerte fijado en: {self.max_trainable_params:,} parámetros")

    def _calculate_param_bounds(self):
        """Calcula min/max params para normalización y establece el techo de seguridad."""
        min_config = {
            'depth': 1, 'initial_filters': self.filters_opts[0], 'kernel_size': (1,1),
            'activation_name': 'ReLU', 'norm_type': 'None', 'dropout_rate': 0.0,
            'use_bias': False, 'pooling_type': 'Max', 'upsample_type': 'BilinearUpsample'
        }

        # Configuración máxima congruente con el nuevo hipercubo expandido
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
            p_max = 35_000_000.0 # Fallback de seguridad estricto
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
            'dropout_rate': self.dropouts_opts[int(variables[5])],
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

    def _create_dataset(self, x, y, custom_batch_size: int, is_training=True):
        """
        Crea el pipeline permitiendo inyectar batch sizes asimétricos (Train vs Val).
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            ds = ds.shuffle(buffer_size=min(128, len(x)))

        ds = ds.batch(custom_batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE) 
        return ds

    def evaluate(self, solution):
        # Limpieza agresiva de VRAM y llamadas al recolector antes de compilar
        K.clear_session()
        gc.collect()

        try:
            self.debugger.start_step('evaluate_solution', {
                'variables': solution.variables.tolist() if isinstance(solution.variables, np.ndarray) else solution.variables,
            })

            # FASE 1: BARRERA GENOTÍPICA (Fail-safe)
            if not self._is_within_bounds(solution.variables):
                print("    [WARN] Arquitectura inválida interceptada en el fail-safe. Penalizando.")
                solution.objectives = np.full(self.n_objectives, np.inf)
                solution.constraints = np.full(self.n_constraints, np.inf)
                solution._invalid_genotype = True 
                return

            config = self.decode_solution(solution.variables)
            print(f"\n--> Evaluando arquitectura: {config}")
            
            # --- CONSTRUCCIÓN DEL FENOTIPO ---
            model = build_unet(self.input_shape, **config)
            raw_params = model.count_params()

            # FASE 2: BARRERA PARAMÉTRICA DE HARDWARE
            if raw_params > self.max_trainable_params:
                print(f"    Arquitectura demasiado masiva ({raw_params:,}). Evitando OOM y penalizando.")
                solution.objectives = np.array([1.0, 1.0]) 
                solution.constraints = np.full(self.n_constraints, np.inf)
                solution.set_metadata(config=config)
                del model
                return

            # FASE 3: CONFIGURACIÓN DEL MOTOR DE ACUMULACIÓN
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

            # Instanciación de los pipelines base
            train_ds = self._create_dataset(self.X_train, self.Y_train, custom_batch_size=self.train_batch_size, is_training=True)
            val_ds = self._create_dataset(self.X_val, self.Y_val, custom_batch_size=self.val_batch_size, is_training=False)

            # --- ENTRENAMIENTO (Optimización de Pesos Físicos) ---
            callbacks = []
            if self.patience > 0:
                # El EarlyStopping siempre monitorea el conjunto val_ds intermedio
                stopper = EarlyStopping(monitor='val_loss', patience=self.patience, mode='min', restore_best_weights=True)
                callbacks.append(stopper)

            start_time = time.time()
            history = model.fit(
                train_ds,
                validation_data=val_ds, 
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=1
            )
            elapsed_time = time.time() - start_time

            # --- FASE 4: BIFURCACIÓN DINÁMICA DE META-VALIDACIÓN ---
            # Comprobación de las condiciones de contorno de los datos de entrada
            if self.X_test is not None and self.Y_test is not None:
                print("    [METODOLOGÍA] X_test detectado. Evaluando sobre el conjunto de Prueba aislado para el fitness de MOEA/D.")
                eval_ds = self._create_dataset(self.X_test, self.Y_test, custom_batch_size=self.val_batch_size, is_training=False)
            else:
                print("    [METODOLOGÍA] X_test es None. Ejecutando fallback: Evaluando sobre el conjunto de Validación (X_val) para el fitness.")
                eval_ds = val_ds

            # Ejecución de la inferencia determinista final con los mejores pesos restaurados
            eval_results = model.evaluate(eval_ds, verbose=1)
            final_val_dice = float(eval_results[1])
            
            # El objetivo evolutivo es estrictamente el complemento del coeficiente de Dice
            obj_dice_loss = float(1.0 - final_val_dice)

            # Normalización lineal del espacio de complejidad paramétrica
            obj_params_norm = float((raw_params - self.z_min_params) / (self.z_max_params - self.z_min_params))
            obj_params_norm = float(np.clip(obj_params_norm, 0.0, 1.0))

            print(f"    Resultados -> Dice Loss: {obj_dice_loss:.4f} | Params Norm: {obj_params_norm:.4f} | Tiempo: {elapsed_time:.1f}s")

            # Inyección segura de objetivos al genotipo activo
            solution.objectives = np.array([obj_dice_loss, float(obj_params_norm)])
            
            # Almacenamiento estructurado de la telemetría generacional
            solution.set_metadata(
                config=config, 
                training_time=elapsed_time,
                epoch=len(history.history['loss']) 
            )

        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
            print(f"    OOM imprevisto en tiempo de ejecución. Penalizando arquitectura de forma acotada.")
            solution.objectives = np.array([1.0, 1.0])
            if 'config' in locals(): solution.set_metadata(config=config)
        
        except Exception as e:
            print(f"    Error general en el hilo de evaluación: {str(e)[:100]} - Penalizando.")
            solution.objectives = np.array([1.0, 1.0])
            if 'config' in locals(): solution.set_metadata(config=config)
        
        finally:
            # --- FASE 5: PURGA ABSOLUTA DE GRADOS DE LIBERTAD EN VRAM ---
            if 'model' in locals(): del model
            if 'train_ds' in locals(): del train_ds
            
            # Si X_test no es None, sabemos con certeza que eval_ds fue 
            # instanciado como un pipeline completamente independiente.
            if 'eval_ds' in locals() and self.X_test is not None: 
                del eval_ds
                
            if 'val_ds' in locals(): del val_ds
            
            K.clear_session()
            gc.collect()
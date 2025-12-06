import keras.backend as K
import gc
import numpy as np

from moead.models import build_unet
from moead.solutions import DLSolution
from moead.problems import Problem

from keras.callbacks import EarlyStopping

from moead.utils import dice_coefficient, dice_loss

class DLProblem(Problem):
    """
    Problema de Optimización de Hiperparámetros para Deep Learning.
    Implementa la estrategia de Lampinen & Zelinka para variables mixtas.
    """
    def __init__(self, 
                 X_train, Y_train, 
                 X_val, Y_val, 
                 X_test=None, Y_test=None,
                 input_shape=(256, 256, 1),
                 batch_size: int = 16,
                 epochs: int = 10):
        
        self.input_shape = input_shape

        # 1. Gestión de Datos (Se cargan una sola vez en memoria)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test

        self.batch_size = batch_size
        self.epochs = epochs
        
        # 2. Definir los Espacios de Búsqueda (Las "Listas" de opciones)
        self.filters_opts = [4, 8, 16, 32]
        self.kernel_opts = [(1,1), (3,3), (5,5)]
        self.act_opts = ['ReLU', 'ELU', 'LeakyReLU', 'GELU', 'Swish']
        self.norm_opts = ['Batch', 'Instance', 'Group', None]
        self.pool_opts = ['Max', 'Average']
        self.upsample_opts = ['TransposeConv', 'BilinearUpsample']
        self.bias_opts = [True, False]
        
        # 3. Definir los Bounds Numéricos para DE
        # El orden de las variables será:
        # [0:Depth, 1:Filters, 2:Kernel, 3:Act, 4:Norm, 5:Dropout, 6:Bias, 7:Pool, 8:UpSample]
        # Usamos rangos continuos [0, N-0.01] para que DE pueda interpolar.
        self._bounds = [
            (1.0, 4.99),                          # 0: Profundidad (Entero 1-4)
            (0.0, len(self.filters_opts) - 0.01), # 1: Filtros (Índice)
            (0.0, len(self.kernel_opts) - 0.01),  # 2: Kernel (Índice)
            (0.0, len(self.act_opts) - 0.01),     # 3: Activación (Índice)
            (0.0, len(self.norm_opts) - 0.01),    # 4: Normalización (Índice)
            (0.0, 0.7),                           # 5: Dropout (Continuo 0.0-0.7)
            (0.0, len(self.bias_opts) - 0.01),    # 6: Bias (Índice)
            (0.0, len(self.pool_opts) - 0.01),    # 7: Pooling (Índice)
            (0.0, len(self.upsample_opts) - 0.01) # 8: UpSampling (Índice)
        ]
        
        self._n_objectives = 2 # Obj 1: Dice Loss (1-Dice), Obj 2: Num Parámetros
        self._n_constraints = 0

        # --- NUEVO: CALCULAR LÍMITES DE NORMALIZACIÓN (Según el paper) ---
        print("Calculando límites teóricos de parámetros (z_min y z_max)...")
        self.z_min_params, self.z_max_params = self._calculate_param_bounds()
        print(f"Límites calculados -> Min: {self.z_min_params:,}, Max: {self.z_max_params:,}")

    def _calculate_param_bounds(self):
        """
        Construye el modelo más pequeño y el más grande posible para obtener 
        los valores de normalización z_min y z_max, tal como indica el paper.
        """
        # 1. Configuración Mínima (Menos profundidad, menos filtros)
        min_config = {
            'depth': 1, # Mínimo del rango
            'initial_filters': 4, # Mínimo del rango
            'kernel_size': (1,1), # Kernel más pequeño (menos pesos)
            'activation_name': 'ReLU',
            'norm_type': None, # Sin params extra
            'dropout_rate': 0.0,
            'use_bias': False, # Menos params
            'pooling_type': 'Max',
            'upsample_type': 'BilinearUpsample' # Upsampling fijo no tiene pesos
        }

        # 2. Configuración Máxima (Más profundidad, más filtros)
        max_config = {
            'depth': 4, # Máximo del rango
            'initial_filters': 32, # Máximo del rango
            'kernel_size': (5,5), # Kernel más grande
            'activation_name': 'PReLU', # Si usaras PReLU tiene params, ReLU no. Asumimos ReLU por defecto.
            'norm_type': 'Batch', # Tiene params entrenables (gamma, beta)
            'dropout_rate': 0.0,
            'use_bias': True,
            'pooling_type': 'Max',
            'upsample_type': 'TransposeConv' # TransposeConv tiene muchos pesos
        }

        # Construir y contar
        K.clear_session()
        model_min = build_unet(self.input_shape, **min_config)
        min_params = model_min.count_params()
        del model_min
        
        K.clear_session()
        # NOTA: Si tu GPU no aguanta el modelo máximo, puedes poner un valor fijo 
        # basado en la literatura (ej. 35,000,000 para U-Nets grandes).
        try:
            model_max = build_unet(self.input_shape, **max_config)
            max_params = model_max.count_params()
            del model_max
        except Exception as e:
            print(f"Advertencia: No se pudo construir el modelo máximo por memoria. Usando estimado.")
            max_params = 20_000_000 # Valor de seguridad (40 millones)

        K.clear_session()
        gc.collect()
        
        return float(min_params), float(max_params)

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
        """
        Convierte el vector numérico de DE en un diccionario de configuración
        legible para construir el modelo.
        """
        # La función int() actúa como truncamiento (floor para positivos),
        # que es lo que sugiere el método de Lampinen & Zelinka.
        config = {
            'depth': int(variables[0]),
            'initial_filters': self.filters_opts[int(variables[1])],
            'kernel_size': self.kernel_opts[int(variables[2])],
            'activation_name': self.act_opts[int(variables[3])],
            'norm_type': self.norm_opts[int(variables[4])],
            'dropout_rate': float(variables[5]), # Se mantiene continuo
            'use_bias': self.bias_opts[int(variables[6])],
            'pooling_type': self.pool_opts[int(variables[7])],
            'upsample_type': self.upsample_opts[int(variables[8])]
        }
        return config

    def evaluate(self, solution):
        # 1. Limpieza preventiva de memoria (CRUCIAL EN NAS)
        K.clear_session()
        gc.collect()

        # 2. Decodificar genotipo
        config = self.decode_solution(solution.variables)
        solution.model_config = config
        print(f"\n--> Evaluando arquitectura: {config}")

        try:
            # 3. Construir Modelo
            model = build_unet(self.input_shape, **config)
            
            # 4. Compilar (Optimizador Adam, Dice Loss)
            model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient])

            # 5. Entrenar
            # Usamos EarlyStopping para ahorrar tiempo (Paper sección 5.2.1)
            # Si el modelo es malo, paramos rápido.
            stopper = EarlyStopping(monitor='val_dice_coefficient', 
                                    patience=3, 
                                    mode='min')

            history = model.fit(
                self.X_train, self.Y_train,
                validation_data=(self.X_val, self.Y_val),
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=[stopper],
                verbose=1 # Silencio para no saturar la consola
            )

            # 6. Calcular Objetivos Reales
            
            # Obj 1: Dice Loss en Validación (Minimizar)
            # Obtenemos el mejor valor logrado durante el entrenamiento
            val_dice_scores = history.history['val_dice_coefficient']
            best_val_dice = max(val_dice_scores)
            obj_dice_loss = 1.0 - best_val_dice  # Eq. 4 del paper

            # Obj 2: Tamaño del Modelo (Minimizar)
            # Contamos parámetros entrenables
            raw_params = model.count_params()  

            obj_params_norm = (float(raw_params) - self.z_min_params) / (self.z_max_params - self.z_min_params)
            
            # Clipping por seguridad (por si acaso floats flotan un poco fuera de rango)
            obj_params_norm = np.clip(obj_params_norm, 0.0, 1.0)

            print(f"    Resultados -> Dice Loss: {obj_dice_loss:.4f} | Params: {obj_params_norm:,}")

            # Asignar a la solución
            solution.objectives = np.array([obj_dice_loss, float(obj_params_norm)])

        except Exception as e:
            print(f"    ERROR: Modelo inválido o OOM. {str(e)}")
            # Penalización severa si el modelo falla (ej. OOM o gradiente explosivo)
            solution.objectives = np.array([1.0, 1e9]) 
        
        finally:
            # 7. Limpieza final agresiva
            if 'model' in locals():
                del model
            K.clear_session()
            gc.collect()
import keras.backend as K
import gc
import numpy as np

from solutions import DLSolution, Solution
from problems import Problem

class DLProblem(Problem):
    """
    Problema de Optimización de Hiperparámetros para Deep Learning.
    Implementa la estrategia de Lampinen & Zelinka para variables mixtas.
    """
    def __init__(self, 
                 X_train, Y_train, 
                 X_val, Y_val, 
                 X_test=None, Y_test=None):
        
        # 1. Gestión de Datos (Se cargan una sola vez en memoria)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        
        # 2. Definir los Espacios de Búsqueda (Las "Listas" de opciones)
        self.filters_opts = [4, 8, 16, 32, 64, 128]
        self.kernel_opts = [(1,1), (3,3), (5,5), (7,7)]
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
            (3.0, 7.99),                          # 0: Profundidad (Entero 3-7)
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
            'activation': self.act_opts[int(variables[3])],
            'normalization': self.norm_opts[int(variables[4])],
            'dropout': float(variables[5]), # Se mantiene continuo
            'use_bias': self.bias_opts[int(variables[6])],
            'pooling': self.pool_opts[int(variables[7])],
            'upsampling': self.upsample_opts[int(variables[8])]
        }
        return config

    def evaluate(self, solution):
        # 1. Decodificar: Obtener la configuración real
        config = self.decode_solution(solution.variables)
        
        # Guardar la config en la solución para el JSONLogger
        solution.model_config = config 
        
        print(f"Evaluando config: {config}")
        
        # --- LÓGICA DE OBJETIVOS BASADA EN PAPER AdaResU-Net ---
        # Objetivo 1: Dice Loss = 1 - Dice Coefficient
        # Objetivo 2: Model Size = Número total de parámetros entrenables
        
        # --- AQUÍ VA TU LÓGICA DE CONSTRUCCIÓN Y ENTRENAMIENTO ---
        
        # Ejemplo conceptual con Keras:
        # K.clear_session()
        # model = build_adaresunet(config) # Tu función que construye la red
        
        # history = model.fit(...)
        
        # val_dice = calculate_dice_score(model, self.X_val, self.Y_val)
        # n_params = model.count_params()
        
        # --- SIMULACIÓN (Ajustada al Paper) ---
        # Simulamos que modelos más profundos y anchos tienen mejor Dice (menor loss)
        # pero mayor costo (más parámetros).
        
        # Factor de capacidad (~ complejidad del modelo)
        capacity_factor = (config['depth'] * 2) * (config['initial_filters'] / 4)
        
        # Simulación de Dice (0 a 1): Mejora con capacidad, empeora con dropout excesivo
        simulated_dice = 0.95 - (1.0 / (capacity_factor + 1)) - (config['dropout'] * 0.05)
        simulated_dice = np.clip(simulated_dice, 0, 0.99) # Clipping lógico
        
        # Objetivo 1: Dice Loss (Minimizar)
        dice_loss = 1.0 - simulated_dice
        
        # Objetivo 2: Número de Parámetros (Minimizar)
        # Aproximación grosera de crecimiento de parámetros en U-Net
        simulated_params = int(1000 * (2 ** config['depth']) * config['initial_filters'])
        
        # ---------------------------------------------------------------
        
        # 2. Asignar Objetivos
        solution.objectives = np.array([dice_loss, simulated_params])
        solution.constraints = np.array([]) # Sin restricciones por ahora
        
        # 3. Limpieza de Memoria (CRÍTICO para GPU)
        # del model
        # gc.collect()
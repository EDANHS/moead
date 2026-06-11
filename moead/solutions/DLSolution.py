import numpy as np
from moead.solutions.Solution import Solution

class DLSolution(Solution):
    """
    Implementación concreta de Solution para problemas de Deep Learning.
    Garantiza el aislamiento genotípico mediante copias explícitas en memoria
    y proporciona mecanismos avanzados para la gestión de metadatos del modelo.
    """
    def __init__(self, 
                 variables: np.ndarray, 
                 n_objectives: int, 
                 n_constraints: int = 0):
        
        # 1. Aislamiento de memoria: Evita la contaminación por referencia entre padres e hijos
        self._variables = np.copy(variables)
        
        # Inicialización de vectores de rendimiento con infinito numérico
        self._objectives = np.full(n_objectives, np.inf)
        self._constraints = np.full(n_constraints, np.inf)
        
        # --- Metadatos Específicos de Deep Learning ---
        self.model_config = {}       # Mapeo legible del fenotipo (ej. {'depth': 3, 'filters': 32})
        self.model_path = None       # Puntero al almacenamiento persistente del modelo (.h5 / .keras)
        self.training_time = 0.0     # Auditoría de coste computacional en segundos
        self.epoch_created = 0       # Registro cronológico generacional
        
        # Flag de control explícito para el flujo defensivo de los operadores
        self._invalid_genotype = False

    # --- Interfaz Contractual (Propiedades de Encapsulamiento) ---
    
    @property
    def objectives(self) -> np.ndarray:
        return self._objectives
    
    @objectives.setter
    def objectives(self, values: np.ndarray):
        self._objectives = np.asarray(values, dtype=float)

    @property
    def variables(self) -> np.ndarray:
        return self._variables
        
    @variables.setter
    def variables(self, values: np.ndarray):
        # Asegura la integridad del genotipo ante reasignaciones externas
        self._variables = np.copy(values)
        
    @property
    def constraints(self) -> np.ndarray:
        return self._constraints
    
    @constraints.setter
    def constraints(self, values: np.ndarray):
        self._constraints = np.asarray(values, dtype=float)

    # --- Métodos de Gestión y Preservación de Linaje ---
    
    def set_metadata(self, config: dict, path: str = None, training_time: float = 0.0, epoch: int = 0):
        """Asigna de forma atómica la telemetría recolectada por el DLProblem."""
        self.model_config = dict(config)  # Copia superficial del diccionario para evitar mutaciones
        self.model_path = path
        self.training_time = float(training_time)
        self.epoch_created = int(epoch)

    def get_config(self) -> dict:
        return getattr(self, 'model_config', {})

    def get_model_path(self) -> str:
        return getattr(self, 'model_path', None)

    def clone(self, keep_performance: bool = True) -> 'DLSolution':
        """
        Genera una copia exacta de la solución.
        
        keep_performance: Si es True, clona los objetivos, restricciones y metadatos
                          (útil para almacenar en el Archive).
                          Si es False, hereda las variables pero resetea el rendimiento
                          a infinito (útil para inicializar hijos antes de evaluar).
        """
        # Crear la nueva instancia con el vector genético aislado
        cloned_solution = type(self)(self._variables, self._objectives.shape[0], self._constraints.shape[0])
        
        if keep_performance:
            cloned_solution.objectives = np.copy(self._objectives)
            cloned_solution.constraints = np.copy(self._constraints)
            cloned_solution.set_metadata(
                config=self.model_config,
                path=self.model_path,
                training_time=self.training_time,
                epoch=self.epoch_created
            )
            cloned_solution._invalid_genotype = self._invalid_genotype
            
        return cloned_solution

    def __repr__(self) -> str:
        # Representación optimizada para logs legibles en consola y depuración
        dice_loss_str = f"{self._objectives[0]:.4f}" if self._objectives[0] != np.inf else "inf"
        params_str = f"{self._objectives[1]:.4f}" if (self._objectives.shape[0] > 1 and self._objectives[1] != np.inf) else "inf"
        return f"DLSolution(DiceLoss={dice_loss_str}, ParamsNorm={params_str}, Depth={self.model_config.get('depth', 'N/A')})"
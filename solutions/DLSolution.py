import numpy as np
from solutions.Solution import Solution

class DLSolution(Solution):
    """
    Implementación concreta de Solution para problemas de Deep Learning.
    Almacena metadatos adicionales como la configuración del modelo.
    """
    def __init__(self, 
                 variables: np.ndarray, 
                 n_objectives: int, 
                 n_constraints: int = 0):
        
        # Datos fundamentales (requeridos por la interfaz Solution)
        self._variables = variables
        self._objectives = np.full(n_objectives, np.inf)
        self._constraints = np.full(n_constraints, np.inf)
        
        # --- Metadatos específicos de DL ---
        self.model_config = {}  # Diccionario legible (ej. {'lr': 0.01, 'layers': 3})
        self.model_path = None  # Ruta al archivo .h5/.pth si se guardó
        self.training_time = 0.0
        self.epoch_created = 0

    # --- Implementación de la Interfaz ---
    
    @property
    def objectives(self) -> np.ndarray:
        return self._objectives
    
    @objectives.setter
    def objectives(self, values: np.ndarray):
        self._objectives = values

    @property
    def variables(self) -> np.ndarray:
        return self._variables
        
    @variables.setter
    def variables(self, values: np.ndarray):
        self._variables = values
        
    @property
    def constraints(self) -> np.ndarray:
        return self._constraints
    
    @constraints.setter
    def constraints(self, values: np.ndarray):
        self._constraints = values

    # --- Métodos Extra (Opcionales pero útiles) ---
    
    def set_metadata(self, config: dict, path: str = None):
        """Helper para guardar metadatos de un solo golpe."""
        self.model_config = config
        self.model_path = path

    def get_config(self):
        """Recupera la configuración de forma segura."""
        return getattr(self, 'model_config', {})

    def get_model_path(self):
        """Recupera la ruta del modelo si existe."""
        return getattr(self, 'model_path', None)

    def __repr__(self):
        return f"DLSolution(obj={self._objectives}, config={self.model_config})"
import numpy as np

class History:
    """Almacena el historial de métricas y arquitecturas de la ejecución."""
    def __init__(self):
        self.z_star_per_gen = []
        self.archive_size_per_gen = []
        self.population_history = []

    def log_generation(self, z_star: np.ndarray, archive_size: int, population: list = None):
        """Registra las métricas de una sola generación."""
        self.z_star_per_gen.append(np.copy(z_star))
        self.archive_size_per_gen.append(archive_size)
        
        # NUEVO: Crear un snapshot ligero de la población actual
        if population is not None:
            pop_snapshot = []
            for sol in population:
                pop_snapshot.append({
                    'variables': sol.variables.tolist() if hasattr(sol.variables, 'tolist') else sol.variables,
                    'objectives': sol.objectives.tolist() if hasattr(sol.objectives, 'tolist') else sol.objectives,
                    'constraints': sol.constraints.tolist() if hasattr(sol.constraints, 'tolist') else sol.constraints,
                    'model_config': getattr(sol, 'model_config', None)
                })
            self.population_history.append(pop_snapshot)

    def get_history(self) -> dict:
        """Devuelve el historial final como un diccionario seguro para JSON/Pickle."""
        return {
            'z_star_per_gen': [z.tolist() if hasattr(z, 'tolist') else z for z in self.z_star_per_gen],
            'archive_size_per_gen': self.archive_size_per_gen,
            'population_history': self.population_history  # NUEVO
        }
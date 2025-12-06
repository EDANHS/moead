import numpy as np

class History:
    """Almacena el historial de métricas de la ejecución."""
    def __init__(self):
        self.z_star_per_gen = []
        self.archive_size_per_gen = []

    def log_generation(self, z_star: np.ndarray, archive_size: int):
        """Registra las métricas de una sola generación."""
        self.z_star_per_gen.append(np.copy(z_star))
        self.archive_size_per_gen.append(archive_size)

    def get_history(self) -> dict:
        """Devuelve el historial final como un diccionario."""
        return {
            'z_star_per_gen': self.z_star_per_gen,
            'archive_size_per_gen': self.archive_size_per_gen
        }
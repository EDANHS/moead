from moead.solutions import Solution
from moead.utils.functions import solution_dominates
import numpy as np

class Archive:
    """
    Almacena el conjunto de soluciones no dominadas (el frente de Pareto).
    
    Implementa un límite de tamaño (max_size) y usa
    Crowding Distance (Distancia de Aglomeración) para podar si se llena.
    Esto soluciona el problema de ralentización en ejecuciones largas.
    """
    def __init__(self, max_size: int = 100):
        """
        Inicializa el archivo.
        
        Args:
            max_size (int): El número máximo de soluciones que el
                            archivo puede almacenar.
        """
        self.solutions: list[Solution] = []
        self.max_size = max_size

    def add(self, new_solution: Solution) -> bool:
        """
        Intenta añadir una nueva solución al archivo.
        
        La lógica es:
        1. Comprueba si la nueva solución es dominada por el archivo.
        2. Si no lo es, elimina todas las soluciones del archivo que 
           son dominadas por la nueva solución.
        3. Añade la nueva solución.
        4. Si el archivo supera 'max_size', lo poda.
        
        Returns:
            bool: True si la solución fue añadida, False en caso contrario.
        """
        
        is_dominated_by_archive = False
        to_remove_indices = [] # Índices de soluciones en el archivo que son dominadas por la nueva
        
        # 1. Comprobar contra el archivo existente
        for i, existing_solution in enumerate(self.solutions):
            if solution_dominates(new_solution, existing_solution):
                # La nueva solución domina a una existente
                to_remove_indices.append(i) 
            elif solution_dominates(existing_solution, new_solution):
                # Una solución existente domina a la nueva
                is_dominated_by_archive = True
                break # La nueva solución es dominada, no añadir
        
        # 2. Si la nueva solución es dominada, no hacer nada
        if is_dominated_by_archive:
            return False

        # 3. Eliminar las soluciones que son dominadas por la nueva
        # (Iterar en reversa para que los índices no se estropeen)
        for i in sorted(to_remove_indices, reverse=True):
            del self.solutions[i]
            
        # 4. Añadir la nueva solución (factible y no dominada)
        self.solutions.append(new_solution)
        
        # 5. Podar el archivo si excede el tamaño máximo
        if self.max_size is not None and len(self.solutions) > self.max_size:
            self._prune()

        return True

    def _prune(self):
        """
        Poda el archivo eliminando la solución menos diversa
        (la que tiene la menor Crowding Distance).
        """
        # Calcular la crowding distance para todas las soluciones en el archivo
        distances = self._calculate_crowding_distance()
        
        # Encontrar el índice de la solución con la MENOR distancia
        idx_to_remove = np.argmin(distances)
        
        # Eliminarla
        del self.solutions[idx_to_remove]

    def _calculate_crowding_distance(self) -> np.ndarray:
        """
        Calcula la crowding distance (distancia de aglomeración)
        para todas las soluciones en self.solutions (lógica de NSGA-II).
        """
        n_solutions = len(self.solutions)
        if n_solutions <= 2:
            # No se puede podar si hay 2 o menos soluciones
            return np.full(n_solutions, np.inf) 

        # Obtener todos los objetivos
        try:
            objectives = np.array([s.objectives for s in self.solutions])
        except (AttributeError, ValueError):
            # Fallback por si las soluciones no son homogéneas (raro)
            return np.full(n_solutions, np.inf)

        n_objectives = objectives.shape[1]
        
        # Inicializar distancias en 0
        distances = np.zeros(n_solutions)
        
        for m in range(n_objectives):
            # Ordenar las soluciones por el objetivo 'm'
            # (obtenemos los índices ordenados)
            sorted_indices = np.argsort(objectives[:, m])
            
            # Asignar distancia infinita a los extremos (para protegerlos)
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # Obtener min y max del objetivo 'm' para normalizar
            f_min = objectives[sorted_indices[0], m]
            f_max = objectives[sorted_indices[-1], m]
            range_m = f_max - f_min
            
            if range_m == 0:
                continue # Omitir si todas las soluciones tienen el mismo valor

            # Calcular la distancia para las soluciones intermedias
            for i in range(1, n_solutions - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]
                next_idx = sorted_indices[i+1]
                
                # Distancia = (valor_siguiente - valor_anterior) / rango_total
                distances[idx] += (objectives[next_idx, m] - objectives[prev_idx, m]) / range_m
                
        return distances

    def get_solutions(self) -> list[Solution]:
        """Devuelve la lista final de soluciones no dominadas."""
        return self.solutions
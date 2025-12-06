from ..crossovers.Crossover import Crossover
from ..evolutionary_operator.EvolutionaryOperator import EvolutionaryOperator
from ..mutations.Mutation import Mutation
import numpy as np
from scipy.spatial.distance import cdist
from ..scalarizations.Scalarization import Scalarization
from ..solutions import Solution
from ..problems.Problem import Problem
from ..utils import Archive, History

class MOEAD:
    """
    Framework MOEA/D 10/10.
    - Incluye 'Archive' para el resultado final (Frente de Pareto).
    - Incluye 'History' para el análisis del proceso.
    - Usa 'EvolutionaryOperator' para la reproducción.
    - Usa 'Scalarization' para el fitness.
    - Implementa Manejo de Restricciones (Reglas de Deb).
    """
    
    def __init__(self, 
                 problem: Problem, 
                 scalarization: Scalarization,
                 evolutionary_op: EvolutionaryOperator,
                 h_divisions: int, 
                 n_neighbors: int, 
                 n_generations: int,
                 n_r: int = 0): # Límite de reemplazo
        
        self.problem = problem
        self.scalarization_op = scalarization
        self.evolutionary_op = evolutionary_op
        
        self.m = problem.n_objectives
        self.h_divisions = h_divisions
        self.n_gen = n_generations

        if n_r == 0:
            self.n_r = n_neighbors
        else:
            self.n_r = n_r 
        
        self.lambda_vectors = self._generate_lambda_vectors(self.m, self.h_divisions)
        self.n_pop = len(self.lambda_vectors)
        self.n_neighbors = min(n_neighbors, self.n_pop)
        self.neighborhoods = self._compute_neighborhoods()
        
        self.population: list[Solution] = []
        self.z_star = np.full(self.m, np.inf)
        
        self.archive = Archive(max_size=h_divisions+1)
        self.history = History()

    def _generate_lambda_vectors(self, m: int, h: int) -> np.ndarray:
        """Genera vectores de peso uniformes (Método SLD)."""
        vectors = []
        def recursive_gen(cv, rh):
            cmi = len(cv)
            if cmi == m - 1:
                vectors.append(np.array(cv + [rh]) / h)
            else:
                for i in range(rh + 1):
                    recursive_gen(cv + [i], rh - i)
        recursive_gen([], h)
        return np.array(vectors)

    def _compute_neighborhoods(self) -> np.ndarray:
        """Calcula los T vecinos más cercanos."""
        distances = cdist(self.lambda_vectors, self.lambda_vectors, 'euclidean')
        return np.argsort(distances, axis=1)[:, :self.n_neighbors]

    def fitness(self, solution: Solution, lambda_vec: np.ndarray) -> float:
        """Calcula el fitness usando la estrategia de escalarización."""
        return self.scalarization_op.execute(solution, lambda_vec, self.z_star)

    def run(self) -> tuple[list[Solution], dict]:
        """
        Ejecuta el bucle de optimización.
        
        Devuelve:
        - (list[Solution]): El frente de Pareto final (del Archivo).
        - (dict): El historial de la ejecución.
        """
        
        print(f"Iniciando MOEA/D (M={self.m}, N={self.n_pop}, T={self.n_neighbors})")
        
        # --- Fase 1: Inicialización ---
        for i in range(self.n_pop):
            sol = self.problem.create_solution() 
            self.problem.evaluate(sol)
            self.population.append(sol)
            self.archive.add(sol)
            
            # Z* solo se actualiza con soluciones FACTIBLES
            v_sol = np.sum(np.maximum(0, sol.constraints))
            if v_sol == 0:
                self.z_star = np.minimum(self.z_star, sol.objectives)
        
        # Log de la Generación 0
        self.history.log_generation(self.z_star, len(self.archive.get_solutions()))
        
        # --- Fase 2: Bucle Evolutivo ---
        for gen in range(self.n_gen):
            if gen % (self.n_gen // 10 or 1) == 0:
                print(f"Generación {gen+1}/{self.n_gen}")

            permutation = np.random.permutation(self.n_pop)
            
            for i in permutation:
                
                child = self.evolutionary_op.execute(
                    i=i,
                    population=self.population,
                    neighborhoods=self.neighborhoods,
                    problem=self.problem
                )
                
                self.problem.evaluate(child)
                
                # Actualizar Z* (solo si el hijo es factible)
                v_child = np.sum(np.maximum(0, child.constraints))
                if v_child == 0:
                    self.z_star = np.minimum(self.z_star, child.objectives)
                
                self.archive.add(child)
                
                shuffled_neighbors = np.random.permutation(self.neighborhoods[i])
                
                replaced_count = 0
                for j in shuffled_neighbors:
                    if replaced_count >= self.n_r:
                        break
                        
                    neighbor = self.population[j]
                    v_neighbor = np.sum(np.maximum(0, neighbor.constraints))
                    
                    # Aplicar Reglas de Dominancia Restringida (Deb's Rules)
                    is_better = False
                    if v_child == 0 and v_neighbor == 0:
                        t_child = self.fitness(child, self.lambda_vectors[j])
                        t_neighbor = self.fitness(neighbor, self.lambda_vectors[j])
                        if t_child < t_neighbor:
                            is_better = True
                    elif v_child == 0 and v_neighbor > 0:
                        is_better = True
                    elif v_child > 0 and v_neighbor == 0:
                        is_better = False
                    else:
                        if v_child < v_neighbor:
                            is_better = True

                    if is_better:
                        self.population[j] = child
                        replaced_count += 1
            
            # --- Log de métricas de la generación ---
            self.history.log_generation(self.z_star, len(self.archive.get_solutions()))
                        
        print("Evolución terminada.")
        
        return self.archive.get_solutions(), self.history.get_history()
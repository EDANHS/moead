from crossovers.Crossover import Crossover
from evolutionary_operator.EvolutionaryOperator import EvolutionaryOperator
from mutations.Mutation import Mutation
import numpy as np
from scipy.spatial.distance import cdist
from scalarizations.Scalarization import Scalarization
from solutions import Solution
from problems import Problem

class MOEAD:
    """
    Implementación concreta y genérica del algoritmo MOEAD.
    Depende solo de las interfaces (Estrategias) que se le inyectan.
    """
    
    def __init__(self, 
                 problem: Problem, 
                 scalarization: Scalarization,
                 evolutionary_op: EvolutionaryOperator,
                 h_divisions: int, 
                 n_neighbors: int, 
                 n_generations: int,
                 n_r: int = 0):
        
        self.problem = problem
        self.scalarization_op = scalarization
        self.evolutionary_op = evolutionary_op
        self.m = problem.n_objectives
        self.h_divisions = h_divisions
        self.n_gen = n_generations

        if n_r <= 0:
            self.n_r = n_neighbors
        else:
            self.n_r = n_r
        
        self.lambda_vectors = self._generate_lambda_vectors(self.m, self.h_divisions)
        self.n_pop = len(self.lambda_vectors)
        self.n_neighbors = min(n_neighbors, self.n_pop)
        self.neighborhoods = self._compute_neighborhoods()
        
        self.population: list[Solution] = []
        self.z_star = np.full(self.m, np.inf)

    def _generate_lambda_vectors(self, m: int, h: int) -> np.ndarray:
        """Genera vectores de peso uniformes (Método SLD)."""
        vectors = []
        def recursive_gen(current_vector, remaining_h):
            current_m_index = len(current_vector)
            if current_m_index == m - 1:
                last_weight = remaining_h
                full_vector = current_vector + [last_weight]
                vectors.append(np.array(full_vector) / h)
            else:
                for i in range(remaining_h + 1):
                    recursive_gen(current_vector + [i], remaining_h - i)
        recursive_gen([], h)
        return np.array(vectors)

    def _compute_neighborhoods(self) -> np.ndarray:
        """Calcula los T vecinos más cercanos para cada vector."""
        distances = cdist(self.lambda_vectors, self.lambda_vectors, 'euclidean')
        return np.argsort(distances, axis=1)[:, :self.n_neighbors]

    def fitness(self, solution: 'Solution', lambda_vec: np.ndarray) -> float:
        """
        Calcula el fitness de una solución usando la estrategia
        de escalarización proporcionada (ej. Tchebycheff, PBI, etc.).
        """
        return self.scalarization_op.execute(solution, lambda_vec, self.z_star)

    def run(self) -> list[Solution]:
        
        print(f"Iniciando MOEA/D (M={self.m}, N={self.n_pop}, T={self.n_neighbors})")
        
        # --- Fase 1: Inicialización (No cambia) ---
        for i in range(self.n_pop):
            sol = self.problem.create_solution() 
            self.problem.evaluate(sol)
            self.population.append(sol)
            self.z_star = np.minimum(self.z_star, sol.objectives)
        
        # --- Fase 2: Bucle Evolutivo (Ahora es genérico) ---
        for gen in range(self.n_gen):
            if gen % (self.n_gen // 10 or 1) == 0:
                print(f"Generación {gen}/{self.n_gen}")

            permutation = np.random.permutation(self.n_pop)
            
            for i in permutation: # Iterar sobre cada subproblema
                
                # --- 1. GENERAR HIJO (Llama a la estrategia inyectada) ---
                child = self.evolutionary_op.execute(
                    i=i,
                    population=self.population,
                    neighborhoods=self.neighborhoods,
                    problem=self.problem
                )
                
                # 2. Evaluar al hijo
                self.problem.evaluate(child)
                
                # 3. Actualizar Z*
                self.z_star = np.minimum(self.z_star, child.objectives)
                
                # 4. Actualización del Vecindario (con límite n_r)
                shuffled_neighbors = np.random.permutation(self.neighborhoods[i])
                
                replaced_count = 0
                for j in shuffled_neighbors:
                    if replaced_count >= self.n_r:
                        break
                        
                    t_child = self.fitness(child, self.lambda_vectors[j])
                    t_neighbor = self.fitness(self.population[j], self.lambda_vectors[j])
                    
                    if t_child < t_neighbor:
                        self.population[j] = child
                        replaced_count += 1
                        
        print("Evolución terminada.")
        return self.population
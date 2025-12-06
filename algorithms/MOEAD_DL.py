from crossovers.Crossover import Crossover
from evolutionary_operator.EvolutionaryOperator import EvolutionaryOperator
from mutations.Mutation import Mutation
from utils.JSONLogger import JSONLogger
import numpy as np
from scipy.spatial.distance import cdist
from scalarizations.Scalarization import Scalarization
from solutions import Solution
from problems import Problem
from utils import Archive, History

import pickle 
import time
import os
import json

class MOEAD_DL:
    """
    Framework MOEA/D optimizado para Deep Learning.
    """
    
    def __init__(self, 
                 problem: Problem, 
                 scalarization: Scalarization,
                 evolutionary_op: EvolutionaryOperator,
                 h_divisions: int, 
                 n_neighbors: int, 
                 n_generations: int,
                 n_r: int = 0, 
                 log_filename: str = "moead_dl_run.json",
                 checkpoint_file: str = "moead_checkpoint.pkl"):
        
        self.problem = problem
        self.scalarization_op = scalarization
        self.evolutionary_op = evolutionary_op
        
        self.m = problem.n_objectives
        self.h_divisions = h_divisions
        self.n_gen = n_generations
        self.checkpoint_file = checkpoint_file
        self.log_filename = log_filename

        if n_r == 0:
            self.n_r = n_neighbors
        else:
            self.n_r = n_r 
        
        self.lambda_vectors = self._generate_lambda_vectors(self.m, self.h_divisions)
        self.n_pop = len(self.lambda_vectors)
        self.n_neighbors = min(n_neighbors, self.n_pop)
        self.neighborhoods = self._compute_neighborhoods()
        
        # Estado Global
        self.population: list[Solution] = []
        self.z_star = np.full(self.m, np.inf)
        self.current_gen = 0
        
        # Componentes
        self.archive = Archive(max_size=self.n_pop) 
        self.history = History() 
        # Inicializamos el logger en None, lo crearemos en run() 
        # para saber si es resume o new
        self.json_logger = None 

    def _generate_lambda_vectors(self, m: int, h: int) -> np.ndarray:
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
        distances = cdist(self.lambda_vectors, self.lambda_vectors, 'euclidean')
        return np.argsort(distances, axis=1)[:, :self.n_neighbors]

    def fitness(self, solution: Solution, lambda_vec: np.ndarray) -> float:
        return self.scalarization_op.execute(solution, lambda_vec, self.z_star)

    def _save_checkpoint(self):
        state = {
            'population': self.population,
            'z_star': self.z_star,
            'current_gen': self.current_gen,
            'archive': self.archive,
            'history': self.history
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        # Además guardamos un resumen legible en JSON junto al pickle
        try:
            summary = {
                'current_gen': int(self.current_gen),
                'z_star': self.z_star.tolist() if hasattr(self.z_star, 'tolist') else list(self.z_star),
                'population': [],
                'history': None
            }
            for sol in self.population:
                try:
                    sol_entry = {
                        'variables': sol.variables.tolist() if hasattr(sol.variables, 'tolist') else sol.variables,
                        'objectives': sol.objectives.tolist() if hasattr(sol.objectives, 'tolist') else sol.objectives,
                        'constraints': sol.constraints.tolist() if hasattr(sol.constraints, 'tolist') else sol.constraints,
                        'model_config': getattr(sol, 'model_config', None)
                    }
                except Exception:
                    sol_entry = {'repr': repr(sol)}
                summary['population'].append(sol_entry)
            # Añadir resumen del history si está disponible
            try:
                if hasattr(self.history, 'get_history'):
                    summary['history'] = self.history.get_history()
                else:
                    summary['history'] = None
            except Exception:
                summary['history'] = None

            json_path = f"{self.checkpoint_file}.json"
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(summary, jf, indent=2, ensure_ascii=False)
        except Exception:
            # No queremos romper el guardado de checkpoint por un error de serialización
            pass
        # print(f"Checkpoint guardado.") # Comentado para no spammear

    def _load_checkpoint(self) -> bool:
        if not os.path.exists(self.checkpoint_file):
            return False
        try:
            with open(self.checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            
            # Validaciones de compatibilidad
            pop = state.get('population')
            z_star = state.get('z_star')
            current_gen = state.get('current_gen')
            archive = state.get('archive')
            history = state.get('history')

            # Comprobar que la población existe y es consistente con el número de subproblemas
            if not isinstance(pop, list) or len(pop) == 0:
                print("Checkpoint encontrado pero la población es inválida. Ignorando checkpoint.")
                return False

            if len(pop) != self.n_pop:
                print(f"Checkpoint incompatible: población guardada={len(pop)} vs esperado={self.n_pop}. Ignorando checkpoint.")
                return False

            # Comprobar z_star
            try:
                if len(z_star) != self.m:
                    print("Checkpoint incompatible: dimensión de z_star no coincide. Ignorando checkpoint.")
                    return False
            except Exception:
                print("Checkpoint corrupto (z_star). Ignorando checkpoint.")
                return False

            # Si todo OK, restaurar estado (con reconstrucciones si es necesario)
            self.population = pop
            self.z_star = np.array(z_star) if not isinstance(z_star, np.ndarray) else z_star
            self.current_gen = int(current_gen) if current_gen is not None else 0

            # Restaurar archive: si viene un objeto compatible, usarlo; si no, crear uno nuevo
            if hasattr(archive, 'get_solutions'):
                self.archive = archive
            else:
                try:
                    # Si archive es una lista de soluciones serializables, reconstruir
                    if isinstance(archive, list):
                        from utils import Archive as _Archive
                        a = _Archive(max_size=self.n_pop)
                        for s in archive:
                            a.add(s)
                        self.archive = a
                    else:
                        self.archive = Archive(max_size=self.n_pop)
                except Exception:
                    self.archive = Archive(max_size=self.n_pop)

            # Restaurar history: puede venir como instancia o dict
            if hasattr(history, 'get_history'):
                self.history = history
            elif isinstance(history, dict):
                # Reconstruir History desde dict
                h = History()
                try:
                    z_hist = history.get('z_star_per_gen', [])
                    arch_hist = history.get('archive_size_per_gen', [])
                    for z, a_size in zip(z_hist, arch_hist):
                        h.z_star_per_gen.append(np.array(z))
                        h.archive_size_per_gen.append(int(a_size))
                    self.history = h
                except Exception:
                    self.history = History()
            else:
                self.history = History()

            print(f"Checkpoint cargado. Reanudando desde Gen {self.current_gen}")
            return True
        except Exception as e:
            print(f"Error cargando checkpoint: {e}. Iniciando desde cero.")
            return False

    def run(self) -> tuple[list[Solution], dict]:
        print(f"Iniciando MOEA/D-DL (M={self.m}, N={self.n_pop})")
        
        # 1. Intentar cargar estado
        resumed = self._load_checkpoint()
        
        # 2. Inicializar Logger (modo resume si cargamos checkpoint)
        self.json_logger = JSONLogger(filename=self.log_filename, resume=resumed)
        
        if not resumed:
            print("Generando población inicial...")
            self.current_gen = 0
            
            for i in range(self.n_pop):
                sol = self.problem.create_solution() 
                self.problem.evaluate(sol)
                self.population.append(sol)
                self.archive.add(sol)
                
                v_sol = np.sum(np.maximum(0, sol.constraints))
                if v_sol == 0:
                    self.z_star = np.minimum(self.z_star, sol.objectives)
                
                print(f"  Solución inicial {i+1}/{self.n_pop} evaluada.")
            
            self.history.log_generation(self.z_star, len(self.archive.get_solutions()))
            self.json_logger.log_generation(0, self.population, self.archive.get_solutions())
            self._save_checkpoint()

        # --- Fase 2: Bucle Evolutivo ---
        start_gen = self.current_gen + 1
        
        for gen in range(start_gen, self.n_gen + 1):
            self.current_gen = gen
            print(f"\n--- Generación {gen}/{self.n_gen} ---")
            start_time = time.time()

            permutation = np.random.permutation(self.n_pop)
            
            for idx, i in enumerate(permutation):
                if idx % 5 == 0: 
                    print(f"  Subproblema {idx+1}/{self.n_pop}...")

                # 1. Generar Hijo
                child = self.evolutionary_op.execute(
                    i=i,
                    population=self.population,
                    neighborhoods=self.neighborhoods,
                    problem=self.problem
                )
                
                # 2. Evaluar
                self.problem.evaluate(child)
                
                # 3. Actualizar Z*
                v_child = np.sum(np.maximum(0, child.constraints))
                if v_child == 0:
                    self.z_star = np.minimum(self.z_star, child.objectives)
                
                # 4. Archive
                self.archive.add(child)
                
                # 5. Vecindario
                shuffled_neighbors = np.random.permutation(self.neighborhoods[i])
                replaced_count = 0
                for j in shuffled_neighbors:
                    if replaced_count >= self.n_r:
                        break
                    
                    neighbor = self.population[j]
                    v_neighbor = np.sum(np.maximum(0, neighbor.constraints))
                    
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
            
            # --- Logs y Checkpoint ---
            self.history.log_generation(self.z_star, len(self.archive.get_solutions()))
            self.json_logger.log_generation(gen, self.population, self.archive.get_solutions())
            self._save_checkpoint()
            
            elapsed = time.time() - start_time
            print(f"  Gen completada en {elapsed:.2f}s. Archive: {len(self.archive.get_solutions())}")
                        
        print("\nEvolución terminada.")
        return self.archive.get_solutions(), self.history.get_history()
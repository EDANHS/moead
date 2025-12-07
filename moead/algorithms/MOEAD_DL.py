import numpy as np
import pickle 
import time
import os
import json
from scipy.spatial.distance import cdist

# Ajusta estos imports según la estructura exacta de carpetas de tu proyecto
# Si están en la misma carpeta o subcarpetas, asegúrate de que apunten bien.
from moead.utils import Archive, History 
from moead.utils.JSONLogger import JSONLogger 

class MOEAD_DL:
    """
    Framework MOEA/D optimizado para Deep Learning con:
    1. Resume Granular (Guarda progreso por individuo).
    2. Logging de Arquitecturas en JSON y Pickle.
    """
    
    def __init__(self, 
                 problem, 
                 scalarization,
                 evolutionary_op,
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
        self.population = [] 
        self.z_star = np.full(self.m, np.inf)
        self.current_gen = 0
        
        # Componentes
        self.archive = Archive(max_size=self.n_pop) 
        self.history = History() 
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

    def fitness(self, solution, lambda_vec: np.ndarray) -> float:
        return self.scalarization_op.execute(solution, lambda_vec, self.z_star)

    def _save_checkpoint(self):
        """
        Guarda el estado completo. 
        Maneja tanto la recuperación binaria (Pickle) como el log legible (JSON).
        """
        # 1. Guardado Binario (CRÍTICO PARA REANUDAR)
        state = {
            'population': self.population,
            'z_star': self.z_star,
            'current_gen': self.current_gen,
            'archive': self.archive,
            'history': self.history
        }
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"Error guardando Pickle: {e}")

        # 2. Guardado JSON (LOG LEGIBLE CON ARQUITECTURAS)
        try:
            summary = {
                'current_gen': int(self.current_gen),
                'z_star': self.z_star.tolist() if hasattr(self.z_star, 'tolist') else list(self.z_star),
                'pop_size': len(self.population),
                'population': [],
                'history': None 
            }
            
            # Desglosamos cada solución para que se vea en el archivo de texto
            for sol in self.population:
                try:
                    sol_entry = {
                        # Variables genéticas
                        'variables': sol.variables.tolist() if hasattr(sol.variables, 'tolist') else sol.variables,
                        # Objetivos
                        'objectives': sol.objectives.tolist() if hasattr(sol.objectives, 'tolist') else sol.objectives,
                        # Restricciones
                        'constraints': sol.constraints.tolist() if hasattr(sol.constraints, 'tolist') else sol.constraints,
                        
                        # --- LA CLAVE: GUARDAR LA CONFIGURACIÓN DEL MODELO ---
                        # Esto asegura que veas "filters": 32, "depth": 4 en el JSON
                        'model_config': getattr(sol, 'model_config', None) 
                    }
                except Exception:
                    sol_entry = {'error': 'Could not serialize solution'}
                
                summary['population'].append(sol_entry)

            # Intentamos añadir historial si existe
            if hasattr(self.history, 'get_history'):
                 summary['history'] = self.history.get_history()

            # Escribir el archivo JSON
            json_path = f"{self.checkpoint_file}.json"
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(summary, jf, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Advertencia: No se pudo guardar el JSON log ({e}), pero el checkpoint .pkl está seguro.")

    def _load_checkpoint(self) -> bool:
        if not os.path.exists(self.checkpoint_file):
            return False
        try:
            print(f"--> Cargando checkpoint: {self.checkpoint_file}")
            with open(self.checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            
            pop = state.get('population')
            z_star = state.get('z_star')
            current_gen = state.get('current_gen')
            archive = state.get('archive')
            history = state.get('history')

            if not isinstance(pop, list):
                print("Checkpoint inválido: Población no es lista.")
                return False

            # Permitir poblaciones parciales (fase de inicialización)
            if len(pop) > self.n_pop:
                print(f"Checkpoint incompatible: demasiados individuos ({len(pop)} > {self.n_pop}).")
                return False
            
            self.population = pop
            self.z_star = np.array(z_star) if not isinstance(z_star, np.ndarray) else z_star
            self.current_gen = int(current_gen) if current_gen is not None else 0

            # Restauración de Archive y History
            self.archive = archive if hasattr(archive, 'get_solutions') else Archive(max_size=self.n_pop)
            self.history = history if hasattr(history, 'get_history') else History()

            print(f"Checkpoint cargado. Individuos: {len(self.population)}/{self.n_pop}. Generación: {self.current_gen}")
            return True
        except Exception as e:
            print(f"Error cargando checkpoint: {e}. Iniciando desde cero.")
            return False

    def run(self):
        print(f"Iniciando MOEA/D-DL (M={self.m}, N={self.n_pop})")
        
        # 1. Intentar cargar estado
        resumed = self._load_checkpoint()
        
        # 2. Inicializar Logger
        self.json_logger = JSONLogger(filename=self.log_filename, resume=resumed)
        
        # 3. Lógica de Inicialización / Reanudación
        current_pop_size = len(self.population)

        # Si estamos en gen 0 y faltan individuos
        if self.current_gen == 0 and current_pop_size < self.n_pop:
            print(f"Generando/Completando población inicial (Inicio: {current_pop_size})...")
            
            # Iteramos desde donde nos quedamos hasta el final
            for i in range(current_pop_size, self.n_pop):
                print(f"  Evaluando Individuo Inicial {i+1}/{self.n_pop}...")
                
                sol = self.problem.create_solution() 
                self.problem.evaluate(sol)
                self.population.append(sol)
                self.archive.add(sol)
                
                # Actualizar Z*
                v_sol = np.sum(np.maximum(0, sol.constraints))
                if v_sol == 0:
                    self.z_star = np.minimum(self.z_star, sol.objectives)
                
                # GUARDADO PARCIAL (El salvavidas de Colab)
                print(f"  --> Guardando checkpoint parcial ({i+1}/{self.n_pop})...")
                self._save_checkpoint()
            
            # Log final de inicialización
            self.history.log_generation(self.z_star, len(self.archive.get_solutions()))
            self.json_logger.log_generation(0, self.population, self.archive.get_solutions())
            self._save_checkpoint()

        elif not resumed:
             pass

        # --- Fase 2: Bucle Evolutivo ---
        start_gen = self.current_gen + 1
        
        if start_gen > self.n_gen:
            print("La evolución ya estaba completa en el checkpoint.")
            return self.archive.get_solutions(), self.history.get_history()

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
                    i=i, population=self.population, neighborhoods=self.neighborhoods, problem=self.problem
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
                    if replaced_count >= self.n_r: break
                    
                    neighbor = self.population[j]
                    v_neighbor = np.sum(np.maximum(0, neighbor.constraints))
                    
                    is_better = False
                    # Tchebycheff / Scalarization Check
                    if v_child == 0 and v_neighbor == 0:
                        t_child = self.fitness(child, self.lambda_vectors[j])
                        t_neighbor = self.fitness(neighbor, self.lambda_vectors[j])
                        if t_child < t_neighbor: is_better = True
                    elif v_child == 0 and v_neighbor > 0: is_better = True
                    elif v_child > 0 and v_neighbor == 0: is_better = False
                    else: 
                        if v_child < v_neighbor: is_better = True

                    if is_better:
                        self.population[j] = child
                        replaced_count += 1
            
            # Logs y Checkpoint por generación
            self.history.log_generation(self.z_star, len(self.archive.get_solutions()))
            self.json_logger.log_generation(gen, self.population, self.archive.get_solutions())
            self._save_checkpoint()
            
            elapsed = time.time() - start_time
            print(f"  Gen completada en {elapsed:.2f}s. Archive: {len(self.archive.get_solutions())}")
                        
        print("\nEvolución terminada.")
        return self.archive.get_solutions(), self.history.get_history()
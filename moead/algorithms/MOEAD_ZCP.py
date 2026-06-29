# Archivo modificado: MOEAD_ZCP.py
import time
import numpy as np
from MOEAD_DL import MOEAD_DL
from moead.utils import JSONLogger
from moead.initializer import ZeroCostWarmup

class MOEAD_ZCP(MOEAD_DL):
    """
    Algoritmo MOEA/D optimizado mediante herencia y composición.
    Delega el filtrado estructural a una clase de Warmup especializada
    antes de comenzar el ciclo de optimización multiobjetivo tradicional.
    """
    def __init__(self, warmup_initializer: ZeroCostWarmup = None, warmup_size=1000, *args, **kwargs):
        """
        :param warmup_initializer: Instancia encargada de poblar la generación cero.
        """
        super().__init__(*args, **kwargs)
        # Si no se provee un inicializador, se instancia uno por defecto (1,000 muestras)
        self.warmup_initializer = warmup_initializer if warmup_initializer else ZeroCostWarmup(warmup_size=warmup_size)

    def run(self):
        print(f"Iniciando MOEA/D-ZCP Acelerado (M={self.m}, N={self.n_pop})")
        
        resumed = self._load_checkpoint()
        self.json_logger = JSONLogger(filename=self.log_filename, resume=resumed)
        current_pop_size = len(self.population)

        # INTERCEPCIÓN DE LA GENERACIÓN CERO MEDIANTE COMPOSICIÓN
        if self.current_gen == 0 and current_pop_size < self.n_pop and not resumed:
            
            # Delegación atómica usando la interfaz .execute()
            elite_population = self.warmup_initializer.execute(self.problem, self.n_pop, self.history)
            
            # Incorporación de la élite a las estructuras del MOEA/D
            for i, sol in enumerate(elite_population):
                print(f"  Formalizando e inicializando metas del Individuo Élite {i+1}/{self.n_pop}...")
                
                # Evaluación subrogada instantánea (vía DLProblemZCP)
                self.problem.evaluate(sol) 
                
                self.population.append(sol)
                self.archive.add(sol.clone(keep_performance=True))
                
                # Actualización inicial del punto ideal de Pareto
                v_sol = np.sum(np.maximum(0, sol.constraints))
                if v_sol == 0:
                    self.z_star = np.minimum(self.z_star, sol.objectives)
                    
            # Logs y persistencia inicial de la ejecución
            self.history.log_generation(self.z_star, len(self.archive.get_solutions()), self.population)
            self.json_logger.log_generation(0, self.population, self.archive.get_solutions())
            self._save_checkpoint()

        elif not resumed:
            pass

        # Bucle Evolutivo Generacional Generalizado
        start_gen = self.current_gen + 1
        if start_gen > self.n_gen:
            print("Evolución ya completada en el estado de restauración.")
            return self.archive.get_solutions(), self.history.get_history()

        for gen in range(start_gen, self.n_gen + 1):
            self.current_gen = gen
            print(f"\n--- Generación {gen}/{self.n_gen} ---")
            start_time = time.time()

            permutation = np.random.permutation(self.n_pop)
            
            for idx, i in enumerate(permutation):
                if idx % 5 == 0: 
                    print(f"  GEN {gen} - Subproblema {idx+1}/{self.n_pop}...")

                # Generación de descendientes (DE / UX decorados por ZCPMoveProposal)
                child = self.evolutionary_op.execute(
                    i=i, population=self.population, neighborhoods=self.neighborhoods,
                    problem=self.problem, debugger=getattr(self.problem, 'debugger', None)
                )

                if getattr(child, '_invalid_genotype', False):
                    continue

                # Evaluación subrogada ultrarrápida (vía DLProblemZCP)
                self.problem.evaluate(child)

                # Actualización de la frontera no dominada
                v_child = np.sum(np.maximum(0, child.constraints))
                if v_child == 0:
                    self.z_star = np.minimum(self.z_star, child.objectives)
                
                self.archive.add(child.clone(keep_performance=True))
                
                # Optimización local en los subproblemas del vecindario (Escalarización de Tchebycheff)
                shuffled_neighbors = np.random.permutation(self.neighborhoods[i])
                replaced_count = 0
                for j in shuffled_neighbors:
                    if replaced_count >= self.n_r: break
                    neighbor = self.population[j]
                    
                    t_child = self.fitness(child, self.lambda_vectors[j])
                    t_neighbor = self.fitness(neighbor, self.lambda_vectors[j])
                    
                    if t_child < t_neighbor:
                        self.population[j] = child.clone(keep_performance=True)
                        replaced_count += 1
            
            self.history.log_generation(self.z_star, len(self.archive.get_solutions()), self.population)
            self.json_logger.log_generation(gen, self.population, self.archive.get_solutions())
            self._save_checkpoint()
            print(f"  Gen completada en {time.time() - start_time:.2f}s. Archive: {len(self.archive.get_solutions())}")
                        
        print("\nEvolución multiobjetivo terminada de forma exitosa.")
        return self.archive.get_solutions(), self.history.get_history()
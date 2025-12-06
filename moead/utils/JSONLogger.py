from datetime import datetime
import numpy as np
import json
import os

class JSONLogger:
    """
    Clase encargada de registrar el progreso del algoritmo en un archivo JSON.
    Guarda los hiperparámetros, métricas y configuración de modelos de la población en cada generación.
    """
    def __init__(self, filename="moead_log.json", resume=False):
        self.filename = filename
        
        # Estructura base
        self.data = {
            "metadata": {
                "algorithm": "MOEA/D",
                "start_time": datetime.now().isoformat(),
                "generations_data": []
            }
        }
        
        # Lógica de carga inteligente:
        # Si resume=True y el archivo existe, cargamos los datos previos.
        # Si resume=False (o no existe), empezamos de cero (sobrescribe o crea).
        if resume and os.path.exists(self.filename):
            try:
                print(f"Cargando log existente: {self.filename}")
                with open(self.filename, 'r') as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                print(f"Advertencia: {self.filename} corrupto. Se creará uno nuevo.")
                self._save()
        else:
            # Si no es resume, inicializamos (y guardamos para crear el archivo)
            self._save()

    def log_generation(self, generation, population, archive, model_configs=None):
        """
        Registra el estado de una generación.
        """
        # Verificar si esta generación ya fue logueada (para evitar duplicados al resumir)
        existing_gens = [g["generation"] for g in self.data["metadata"]["generations_data"]]
        if generation in existing_gens:
            # Si ya existe, actualizamos esa entrada en lugar de añadir una nueva
            # (Útil si reiniciamos a mitad de una generación)
            gen_index = existing_gens.index(generation)
            # Podríamos sobrescribir, pero por simplicidad en logs secuenciales,
            # asumiremos que si llamamos a log, es datos nuevos/finales de esa gen.
            # Para logs append-only, simplemente retornamos o borramos la entrada vieja.
            del self.data["metadata"]["generations_data"][gen_index]

        gen_data = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "archive_size": len(archive),
            "solutions": []
        }

        for i, sol in enumerate(population):
            # Obtener configuración del modelo
            config = {}
            if model_configs and i < len(model_configs):
                config = model_configs[i]
            
            if not config and hasattr(sol, 'model_config'):
                config = getattr(sol, 'model_config')

            sol_data = {
                "id": i, 
                "variables": sol.variables.tolist() if isinstance(sol.variables, np.ndarray) else sol.variables,
                "objectives": sol.objectives.tolist() if isinstance(sol.objectives, np.ndarray) else sol.objectives,
                "constraints": sol.constraints.tolist() if isinstance(sol.constraints, np.ndarray) else sol.constraints,
                "model_config": config 
            }
            gen_data["solutions"].append(sol_data)

        self.data["metadata"]["generations_data"].append(gen_data)
        self._save()

    def _save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=4)
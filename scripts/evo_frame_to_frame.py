"""
Herramienta de Cinemática Evolutiva Discreta (Frame-by-Frame Generator).
Utiliza el isomorfismo matemático de la clase 'Archive' original para extraer
la frontera de Pareto exacta de cada instante temporal. Genera una secuencia
de imágenes a escala estandarizada (1:1) listas para análisis dinámico.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ==========================================
# 1. ISOMORFISMO MATEMÁTICO (Lógica Original)
# ==========================================
class MockSolution:
    """Clase proxy ligera para emular la estructura de la solución original."""
    def __init__(self, objectives, constraints):
        self.objectives = np.array(objectives)
        self.constraints = np.array(constraints)

def solution_dominates(sol_A: MockSolution, sol_B: MockSolution) -> bool:
    """
    Filtro de dominancia nativo extraído directamente de la arquitectura del orquestador.
    Evalúa la factibilidad (constraints) antes que los objetivos.
    """
    v_A = np.sum(np.maximum(0, sol_A.constraints))
    v_B = np.sum(np.maximum(0, sol_B.constraints))

    if v_A > 0 and v_B == 0:
        return False
    if v_A == 0 and v_B > 0:
        return True 
    if v_A > 0 and v_B > 0:
        return v_A < v_B

    a_obj = sol_A.objectives
    b_obj = sol_B.objectives
    
    if np.any(a_obj > b_obj):
        return False 
        
    if np.any(a_obj < b_obj):
        return True 
        
    return False

def get_native_pareto_front(snapshot: list) -> np.ndarray:
    """
    Aplica la lógica del Archive para extraer la frontera no dominada 
    de una generación específica.
    """
    solutions = []
    for sol_data in snapshot:
        obj = sol_data.get('objectives')
        const = sol_data.get('constraints')
        if obj is not None and const is not None:
            solutions.append(MockSolution(obj, const))
            
    if not solutions:
        return np.array([])

    non_dominated_indices = []
    
    for i, current_sol in enumerate(solutions):
        is_dominated = False
        for j, other_sol in enumerate(solutions):
            if i != j:
                if solution_dominates(other_sol, current_sol):
                    is_dominated = True
                    break
        if not is_dominated:
            non_dominated_indices.append(i)
            
    # Extraemos solo los objetivos de las soluciones que sobrevivieron al filtro
    pareto_front = np.array([solutions[i].objectives for i in non_dominated_indices])
    return pareto_front

# ==========================================
# 2. MOTOR DE RENDERIZADO CINEMÁTICO
# ==========================================
def generate_frames(history_json: str, output_dir: str, target_max_gen: int):
    print(f"--> [INICIO] Extrayendo telemetría temporal desde: {history_json}")
    
    try:
        with open(history_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR CRÍTICO] Archivo no encontrado: {history_json}")
        return

    pop_history = data.get('history', {}).get('population_history', [])
    if not pop_history:
        print("[ERROR CRÍTICO] El historial está vacío.")
        return

    target_max_gen = min(target_max_gen, len(pop_history) - 1)
    
    # 2.1 Estandarización de Escala (Cálculo de Límites Globales)
    # Buscamos el máximo y mínimo absoluto de toda la historia para anclar los ejes
    all_x = []
    all_y = []
    for gen in range(target_max_gen + 1):
        front = get_native_pareto_front(pop_history[gen])
        if len(front) > 0:
            all_x.extend(front[:, 0])
            all_y.extend(front[:, 1])
            
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Añadimos un pequeño margen geométrico (padding visual)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"--> [PREPARACIÓN] Ejes fijados matemáticamente. X:[{x_min:.4f}, {x_max:.4f}], Y:[{y_min:.4f}, {y_max:.4f}]")
    print(f"--> [RENDERIZADO] Generando {target_max_gen + 1} fotogramas cinemáticos...")

    # 2.2 Generación de Fotogramas Discretos
    for gen in range(target_max_gen + 1):
        front = get_native_pareto_front(pop_history[gen])
        
        if len(front) == 0:
            continue
            
        # Ordenamiento para trazar la línea continua
        sorted_indices = np.argsort(front[:, 0])
        front = front[sorted_indices]

        plt.figure(figsize=(9, 6))
        
        # Anclaje estricto de la escala 1:1
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)

        # Renderizado de la topología
        plt.plot(front[:, 0], front[:, 1], marker='o', markersize=6, linewidth=2, 
                 color='#1f77b4', label=f'Frente de Pareto (Gen {gen})')
        plt.fill_between(front[:, 0], front[:, 1], y_max + y_margin, color='#1f77b4', alpha=0.1)

        plt.xlabel('Dice Loss (Minimizar)', fontsize=12, fontweight='bold')
        plt.ylabel('Complejidad Computacional (Norm Params)', fontsize=12, fontweight='bold')
        plt.title(f'Estado Topológico Evolutivo: Generación {gen}\nAlgoritmo MOEAD-DL', fontsize=14)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper right', fontsize=11)
        plt.tight_layout()
        
        # Guardar con formato de ceros a la izquierda (frame_00.png, frame_01.png...)
        filename = os.path.join(output_dir, f"frame_{gen:02d}.png")
        plt.savefig(filename, dpi=200)
        plt.close()
        
        print(f"    -> Fotograma {gen:02d} renderizado ({len(front)} arquitecturas élite).")

    print(f"--> [ÉXITO SECUENCIAL] Todos los fotogramas guardados en el directorio: {output_dir}")

if __name__ == '__main__':
    # Configuración de Rutas
    TARGET_GENERATION = 20 # Ajusta al horizonte truncado de tu tesis
    
    HISTORY_JSON = "uniform_moead_dl_checkpoint_ctv.pkl.json"
    
    # Directorio dedicado para que las imágenes no ensucien la raíz
    OUTPUT_DIRECTORY = "frames_evolution/"
    
    generate_frames(HISTORY_JSON, OUTPUT_DIRECTORY, TARGET_GENERATION)
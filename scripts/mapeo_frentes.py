"""
Módulo de Superposición Topológica y Auditoría de Frentes (Multi-Algo Pareto Plotter).
Extrae la población de N algoritmos en una generación temporal específica (target_gen),
aplica el isomorfismo matemático de dominancia para aislar élites de subóptimos,
y renderiza una radiografía completa del hiperespacio de búsqueda utilizando 
una jerarquía cromática de profundidad (Z-Order Layering).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. ISOMORFISMO MATEMÁTICO (Filtro Nativo)
# ==========================================
class MockSolution:
    """Clase proxy ligera para emular la estructura transaccional original."""
    def __init__(self, objectives, constraints):
        self.objectives = np.array(objectives)
        self.constraints = np.array(constraints)

def solution_dominates(sol_A: MockSolution, sol_B: MockSolution) -> bool:
    """Filtro de dominancia estricto priorizando factibilidad."""
    v_A = np.sum(np.maximum(0, sol_A.constraints))
    v_B = np.sum(np.maximum(0, sol_B.constraints))

    if v_A > 0 and v_B == 0: return False
    if v_A == 0 and v_B > 0: return True 
    if v_A > 0 and v_B > 0: return v_A < v_B

    a_obj = sol_A.objectives
    b_obj = sol_B.objectives
    
    if np.any(a_obj > b_obj): return False 
    if np.any(a_obj < b_obj): return True 
    return False

def get_population_split(snapshot: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica la lógica de dominancia para bifurcar la población temporal en dos 
    conjuntos topológicos: El Frente de Pareto y el Conjunto de Soluciones Dominadas.
    """
    solutions = []
    for sol_data in snapshot:
        obj = sol_data.get('objectives')
        const = sol_data.get('constraints')
        if obj is not None and const is not None:
            solutions.append(MockSolution(obj, const))
            
    if not solutions:
        return np.array([]), np.array([])

    non_dominated_indices = []
    dominated_indices = []
    
    for i, current_sol in enumerate(solutions):
        is_dominated = False
        for j, other_sol in enumerate(solutions):
            if i != j:
                if solution_dominates(other_sol, current_sol):
                    is_dominated = True
                    break
        if not is_dominated:
            non_dominated_indices.append(i)
        else:
            dominated_indices.append(i)
            
    pareto_front = np.array([solutions[i].objectives for i in non_dominated_indices])
    dominated_set = np.array([solutions[i].objectives for i in dominated_indices])
    
    return pareto_front, dominated_set

# ==========================================
# 2. EXTRACCIÓN HISTÓRICA (DATA FETCHING)
# ==========================================
def extract_generation_population(history_json: str, target_gen: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Lee el archivo de telemetría episódica, viaja a la generación solicitada 
    (o a la última disponible si hubo Early Truncation) y extrae la topología.
    """
    try:
        with open(history_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR CRÍTICO] Archivo no encontrado: {history_json}")
        return np.array([]), np.array([]), 0

    pop_history = data.get('history', {}).get('population_history', [])
    if not pop_history:
        print(f"[ERROR CRÍTICO] El historial está vacío en {history_json}.")
        return np.array([]), np.array([]), 0

    # Truncamiento de seguridad: Si la Gen pedida es mayor a la existente, toma la última
    actual_gen = min(target_gen, len(pop_history) - 1)
    
    front, dominated = get_population_split(pop_history[actual_gen])
    return front, dominated, actual_gen

# ==========================================
# 3. MOTOR DE RENDERIZADO VISUAL
# ==========================================
def plot_superimposed_fronts(config: dict, target_gen: int, output_img: str):
    print(f"--> [INICIO] Orquestando Superposición Topológica para la Generación {target_gen}...")
    
    plt.figure(figsize=(11, 7))
    
    # Paletas cromáticas de alta jerarquía (Soporte para N Algoritmos)
    # Formato: (Color_Élite, Color_Dominado, Marcador)
    THEMES = [
        ('#1f77b4', '#a0c4df', 'o'), # UX: Azul Oscuro / Celeste Claro
        ('#ff7f0e', '#ffbb78', 's'), # DE: Naranja Fuerte / Naranja Suave
        ('#2ca02c', '#98df8a', '^'), # Extra: Verde Oscuro / Verde Claro
        ('#d62728', '#ff9896', 'D')  # Extra: Rojo Oscuro / Rojo Claro
    ]
    
    for idx, (algo_name, filepath) in enumerate(config.items()):
        front, dominated, actual_gen = extract_generation_population(filepath, target_gen)
        
        if len(front) == 0 and len(dominated) == 0:
            print(f"  -> [WARN] Sin datos topológicos para {algo_name}. Omitiendo.")
            continue
            
        print(f"  -> [Procesado] {algo_name}: {len(front)} Élites | {len(dominated)} Dominadas (Gen recuperada: {actual_gen})")
        
        theme = THEMES[idx % len(THEMES)]
        color_elite, color_dominated, marker = theme
        
        # 1° CAPA (FONDO): Exploración Subóptima (Z-Order 1)
        if len(dominated) > 0:
            plt.scatter(dominated[:, 0], dominated[:, 1], 
                        color=color_dominated, alpha=0.4, s=30, 
                        edgecolors='white', linewidths=0.5, 
                        label=f'{algo_name} (Exploración Dominada)', zorder=1)

        # 2° CAPA (FRENTE): Topología Élite (Z-Order 3)
        if len(front) > 0:
            # Ordenamiento geométrico por Eje X para evitar líneas cruzadas
            sorted_indices = np.argsort(front[:, 0])
            front = front[sorted_indices]
            
            # Trazado de la línea con marcadores sobre la nube de puntos
            plt.plot(front[:, 0], front[:, 1], 
                     marker=marker, markersize=8, linewidth=2.5, 
                     color=color_elite, label=f'{algo_name} (Frente de Pareto)', 
                     zorder=3, alpha=0.95)

    # Parametrización del lienzo y jerarquía de la información
    plt.xlabel('Dice Loss (Minimizar)', fontsize=12, fontweight='bold')
    plt.ylabel('Complejidad Computacional (Norm Params)', fontsize=12, fontweight='bold')
    plt.title(f'Radiografía del Espacio de Búsqueda NAS (Generación {target_gen})\nComparativa de Dispersión: Élites vs. Soluciones Subóptimas', fontsize=14, pad=15)
    
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # Leyenda optimizada para no tapar los datos
    plt.legend(loc='upper right', fontsize=10, shadow=True, framealpha=0.95)
    plt.tight_layout()
    
    Path(output_img).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img, dpi=300)
    plt.close()
    
    print(f"--> [ÉXITO METODOLÓGICO] Gráfico de frentes superpuestos generado en: {output_img}")

# ==========================================
# 4. ORQUESTADOR DE PRESENTACIÓN
# ==========================================
if __name__ == '__main__':
    # ---------------------------------------------------------
    # CONFIGURACIÓN DEL EXPERIMENTO
    # ---------------------------------------------------------
    
    # La generación temporal que deseas observar. 
    # Si un algoritmo no llegó, mostrará su última generación disponible.
    TARGET_GENERATION = 8 
    
    # Diccionario explícito de algoritmos (Apunta a los .pkl.json)
    ALGORITHMS_CONFIG = {
        "Cruce Uniforme (UX+Tcheb)": "archivos_cache/uniform_moead_dl_checkpoint_ctv.pkl.json",
        "Evolución Diferencial (DE+PBI)": "archivos_cache/de_moead_dl_checkpoint_ctv.pkl.json"
    }
    
    OUTPUT_IMAGE = f"plot_04_superimposed_fronts_gen{TARGET_GENERATION}.png"
    
    plot_superimposed_fronts(ALGORITHMS_CONFIG, TARGET_GENERATION, OUTPUT_IMAGE)
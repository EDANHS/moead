import json
import numpy as np
from pathlib import Path

def get_non_dominated_front(objectives: np.ndarray) -> np.ndarray:
    """Filtro matemático de Dominancia de Pareto estricto (Tu función original)."""
    is_efficient = np.ones(objectives.shape[0], dtype=bool)
    for i, c in enumerate(objectives):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(objectives[is_efficient] < c, axis=1)
            is_efficient[i] = True
            is_efficient[is_efficient] = np.logical_not(
                np.all(objectives[is_efficient] >= c, axis=1) & 
                np.any(objectives[is_efficient] > c, axis=1)
            )
    return is_efficient

def extract_compromise_solutions(history_json: str, target_gen: int, output_json: str):
    print(f"--> [INICIO] Analizando compromisos para la Generación {target_gen}...")
    
    # 1. Carga del archivo histórico
    try:
        with open(history_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Archivo no encontrado: {history_json}")
        return

    pop_history = data.get('history', {}).get('population_history', [])
    if not pop_history or target_gen >= len(pop_history):
        print(f"[ERROR] La generación {target_gen} no existe o el JSON está vacío.")
        return

    # 2. Extracción de soluciones válidas de la Gen X
    snapshot = pop_history[target_gen]
    valid_solutions = []
    objectives_list = []
    
    for sol in snapshot:
        obj = sol.get('objectives')
        if obj is not None and len(obj) >= 2:
            valid_solutions.append(sol)
            objectives_list.append(obj)
            
    if not objectives_list:
        print(f"[ERROR] No se encontraron objetivos válidos en la Gen {target_gen}.")
        return
        
    objectives = np.array(objectives_list)
    
    # 3. Filtrar por Frontera de Pareto
    pareto_mask = get_non_dominated_front(objectives)
    pareto_indices = np.where(pareto_mask)[0]
    pareto_objectives = objectives[pareto_mask]
    
    print(f"--> [INFO] Soluciones totales en Gen {target_gen}: {len(objectives)} | En el frente de Pareto: {len(pareto_objectives)}")

    # 4. Identificación de los 3 perfiles de compromiso
    # Objetivo 0: Dice Loss (Minimizar) -> Eje X
    # Objetivo 1: Complejidad/Parámetros (Minimizar) -> Eje Y
    
    # Compromiso 1: Menor Dice Loss (por ende, mayor complejidad en la frontera)
    idx_best_dice_rel = np.argmin(pareto_objectives[:, 0])
    idx_best_dice_abs = pareto_indices[idx_best_dice_rel]
    
    # Compromiso 2: Menor Complejidad (por ende, mayor Dice Loss en la frontera)
    idx_best_comp_rel = np.argmin(pareto_objectives[:, 1])
    idx_best_comp_abs = pareto_indices[idx_best_comp_rel]
    
    # Compromiso 3: Solución balanceada (Knee Point mediante distancia Euclídea Normalizada)
    f1 = pareto_objectives[:, 0]
    f2 = pareto_objectives[:, 1]
    
    max_f1, min_f1 = np.max(f1), np.min(f1)
    max_f2, min_f2 = np.max(f2), np.min(f2)
    
    # Normalización min-max para evitar sesgos por diferencias de magnitudes
    f1_norm = (f1 - min_f1) / (max_f1 - min_f1) if max_f1 != min_f1 else np.zeros_like(f1)
    f2_norm = (f2 - min_f2) / (max_f2 - min_f2) if max_f2 != min_f2 else np.zeros_like(f2)
    
    # Distancia al punto utópico/ideal (0, 0)
    distances_to_ideal = np.sqrt(f1_norm**2 + f2_norm**2)
    idx_knee_rel = np.argmin(distances_to_ideal)
    idx_knee_abs = pareto_indices[idx_knee_rel]

    # 5. Consolidación de resultados
    compromise_portfolio = {
        "meta": {
            "source_history": history_json,
            "target_generation": target_gen,
            "total_pareto_solutions_in_gen": len(pareto_objectives)
        },
        "compromises": {
            "best_dice_loss_extreme": {
                "descripcion": "Mayor complejidad computacional, pero el Dice Loss más bajo (Élite de Rendimiento).",
                "datos_red": valid_solutions[idx_best_dice_abs]
            },
            "best_complexity_extreme": {
                "descripcion": "Mayor Dice Loss, pero la menor complejidad computacional (Élite de Eficiencia/Ligera).",
                "datos_red": valid_solutions[idx_best_comp_abs]
            },
            "knee_point_balanced": {
                "descripcion": "Solución de compromiso óptimo (Knee Point) usando distancia euclidiana normalizada al origen.",
                "datos_red": valid_solutions[idx_knee_abs]
            }
        }
    }

    # 6. Almacenamiento en JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(compromise_portfolio, f, indent=4, ensure_ascii=False)
        
    print(f"--> [ÉXITO] Portafolio de arquitectura consolidado impecablemente en: {output_json}")

if __name__ == '__main__':
    # ==========================================
    # CONFIGURACIÓN DEL PIPELINE DE EXTRACCIÓN
    # ==========================================
    GEN_X = 25  # Define aquí la generación exacta que necesitas auditar
    HISTORY_JSON = "archivos_cache/uniform_moead_dl_checkpoint_ctv.pkl.json"
    OUTPUT_JSON = f"pareto_compromises_gen_{GEN_X}_uniform.json"
    
    extract_compromise_solutions(HISTORY_JSON, GEN_X, OUTPUT_JSON)
"""
Herramienta de Análisis Multi-Temporal y Trazabilidad Evolutiva.
Lee el historial poblacional desde la clase History (JSON), extrae la frontera
no dominada de cada generación de forma independiente, y renderiza la trayectoria
de convergencia cromática del motor MOEA/D.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

def get_non_dominated_front(objectives: np.ndarray) -> np.ndarray:
    """Filtro matemático de Dominancia de Pareto estricto."""
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

def extract_and_plot_evolution(history_json: str, target_max_gen: int, output_img: str):
    print(f"--> [INICIO] Arrancando Motor de Trazabilidad Evolutiva: {history_json}")
    
    try:
        with open(history_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR CRÍTICO] Archivo de historial no encontrado: {history_json}")
        return

    pop_history = data.get('history', {}).get('population_history', [])
    
    if not pop_history:
        print("[ERROR CRÍTICO] El JSON no contiene 'population_history'.")
        return

    # Ajuste de seguridad si target_max_gen excede lo disponible
    max_gen_available = len(pop_history) - 1
    target_max_gen = min(target_max_gen, max_gen_available)
    
    print(f"--> [EXTRACCIÓN] Procesando fronteras temporales desde Gen 0 hasta Gen {target_max_gen}...")

    # Contenedores para almacenar la evolución topológica
    fronts_by_gen = {}

    for gen in range(target_max_gen + 1):
        snapshot = pop_history[gen]
        objectives = []
        
        for sol in snapshot:
            obj = sol.get('objectives')
            if obj is not None and len(obj) >= 2:
                objectives.append(obj)
                
        objectives = np.array(objectives)
        
        # Saltamos si hubo un error en esta generación
        if len(objectives) == 0:
            continue
            
        # 1. Filtro Ortogonal: Extraemos solo las élites de ESTA generación temporal
        pareto_mask = get_non_dominated_front(objectives)
        pareto_front = objectives[pareto_mask]
        
        # 2. Ordenamiento Geométrico (Ordenar por Eje X para que las líneas se dibujen bien)
        sorted_indices = np.argsort(pareto_front[:, 0])
        pareto_front = pareto_front[sorted_indices]
        
        fronts_by_gen[gen] = pareto_front

    # ==========================================
    # FASE DE AUDITORÍA VISUAL (RENDERIZADO)
    # ==========================================
    print("--> [AUDITORÍA] Renderizando Mapeo Cromático de Convergencia...")
    plt.figure(figsize=(10, 7))
    
    # Seleccionamos un colormap secuencial (viridis, plasma, o coolwarm)
    # Viridis va de Púrpura oscuro (Gen 0) a Amarillo brillante (Gen Final)
    cmap = cm.viridis 
    
    # Graficamos generación por generación para mostrar el "empuje" evolutivo
    for gen, front in fronts_by_gen.items():
        # Normalizamos la generación actual entre 0 y 1 para el color
        color_intensity = gen / float(target_max_gen) if target_max_gen > 0 else 1.0
        color = cmap(color_intensity)
        
        # Para evitar saturar la leyenda, solo la agregamos a ciertas generaciones clave
        label = f'Gen {gen}' if gen in [0, target_max_gen//2, target_max_gen] else "_nolegend_"
        
        # Si es la última generación (El Frente Final), le damos énfasis visual absoluto
        if gen == target_max_gen:
            plt.plot(front[:, 0], front[:, 1], marker='o', markersize=7, linewidth=2.5, 
                     color='red', label=f'Frente Final (Gen {gen})', zorder=5)
        else:
            # Las generaciones pasadas se grafican como puntos translúcidos y líneas tenues
            plt.plot(front[:, 0], front[:, 1], marker='.', markersize=4, linewidth=0.8, 
                     color=color, alpha=0.4, label=label, zorder=2)

    # Configuración de los ejes y jerarquía de la información
    plt.xlabel('Dice Loss (Minimizar)', fontsize=12, fontweight='bold')
    plt.ylabel('Complejidad Computacional (Norm Params)', fontsize=12, fontweight='bold')
    plt.title('Trayectoria de Convergencia Evolutiva (MOEAD-DL)\nEvolución del Frente de Pareto en el Hiperespacio NAS', fontsize=14)
    
    # Barra de color (Colorbar) para indicar el paso del tiempo
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=target_max_gen))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Progreso Temporal (Generación)', rotation=270, labelpad=15, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    
    plt.savefig(output_img, dpi=300)
    plt.close()
    
    print(f"--> [ÉXITO] Evolución del Frente de Pareto generada impecablemente en: {output_img}")

if __name__ == '__main__':
    # ==========================================
    # CONFIGURACIÓN DEL PIPELINE
    # ==========================================
    TARGET_GENERATION = 8 # El horizonte temporal truncado para tu tesis
    
    # Ruta a tu historial serializado (Ajusta si es necesario)
    HISTORY_JSON = "archivos_cache/uniform_moead_dl_checkpoint_ctv.pkl.json"
    
    # Ruta de salida del gráfico
    OUTPUT_IMAGE = "pareto_evolution.png"
    
    Path(OUTPUT_IMAGE).parent.mkdir(parents=True, exist_ok=True)
    
    extract_and_plot_evolution(HISTORY_JSON, TARGET_GENERATION, OUTPUT_IMAGE)
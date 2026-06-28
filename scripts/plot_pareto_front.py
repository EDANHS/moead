import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURACIÓN ---
CHECKPOINT_FILE = "moead_dl_checkpoint.pkl.json"
OUTPUT_FILE = "frente_pareto_final_clasificado.png"

# Límites del gráfico
X_MAX_LIMIT = 0.07
Y_MAX_LIMIT = 0.007

# --- FUNCIÓN DE UTILIDAD: CLASIFICACIÓN DE DOMINANCIA ---

def is_dominated(point_a, objectives):
    """
    Verifica si el punto_a (una lista/tupla de objetivos) es dominado 
    por cualquier otro punto en la lista 'objectives'.
    (Asume problemas de minimización).
    """
    for point_b in objectives:
        # El punto_b domina al punto_a si:
        # 1. point_b es mejor o igual en ambos objetivos (f1 y f2)
        # 2. point_b es estrictamente mejor en al menos uno.
        
        # Dominancia (para minimización):
        cond1 = point_b[0] <= point_a[0]  # B es mejor o igual en el objetivo 1
        cond2 = point_b[1] <= point_a[1]  # B es mejor o igual en el objetivo 2
        
        # Estrictamente mejor en al menos uno
        cond3 = point_b[0] < point_a[0] or point_b[1] < point_a[1]
        
        if cond1 and cond2 and cond3:
            return True  # point_a es dominado por point_b
            
    return False # point_a NO es dominado por ningún otro punto en la lista

# --- FUNCIÓN PRINCIPAL DE GRAFICADO ---

def plot_final_pareto_front():
    """Carga los datos del checkpoint, clasifica y grafica la población final."""
    print(f"--> Leyendo archivo checkpoint: {CHECKPOINT_FILE}")
    
    # 1. Cargar el archivo JSON
    try:
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Archivo {CHECKPOINT_FILE} no encontrado. Asegúrate de la ruta y la extensión.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: No se pudo decodificar el archivo JSON.")
        return

    # 2. Extraer los datos de la población final
    population = data.get('population', [])
    current_gen = data.get('current_gen', 'N/A')
    
    if not population:
        print("ERROR: No se encontró la clave 'population' o está vacía.")
        return

    # Extraer y filtrar los objetivos (solo soluciones con al menos 2 objetivos)
    all_objectives = [s['objectives'] for s in population if s.get('objectives') and len(s['objectives']) >= 2]
    
    if not all_objectives:
        print("ERROR: No se encontraron objetivos válidos en la población.")
        return

    # 3. Clasificar las soluciones
    
    # Listas para separar los puntos
    pareto_front_loss = []
    pareto_front_params = []
    dominated_loss = []
    dominated_params = []
    
    print(f"--> Clasificando {len(all_objectives)} soluciones...")
    
    for obj in all_objectives:
        if is_dominated(obj, all_objectives):
            dominated_loss.append(obj[0])
            dominated_params.append(obj[1])
        else:
            pareto_front_loss.append(obj[0])
            pareto_front_params.append(obj[1])

    print(f"    - Soluciones No Dominadas (Frente de Pareto): {len(pareto_front_loss)}")
    print(f"    - Soluciones Dominadas: {len(dominated_loss)}")
    
    # 4. Determinar el punto Z_star (punto ideal/referencia)
    z_star = data.get('z_star', None)

    # 5. Generar el gráfico
    plt.figure(figsize=(9, 7))
    
    # Graficar las soluciones DOMINADAS (primero para que queden debajo)
    plt.scatter(dominated_loss, dominated_params, 
                s=30, alpha=0.5, color='lightgray', edgecolors='none', 
                label=f'Dominadas (N={len(dominated_loss)})')

    # Graficar las soluciones NO DOMINADAS (Frente de Pareto)
    plt.scatter(pareto_front_loss, pareto_front_params, 
                s=50, alpha=0.8, color='blue', edgecolors='black', linewidths=0.7, 
                label=f'Frente de Pareto (N={len(pareto_front_loss)})')

    # Graficar el punto ideal Z_star si existe
    if z_star and len(z_star) >= 2:
        plt.scatter(z_star[0], z_star[1], 
                    s=150, marker='*', color='gold', edgecolors='black', linewidths=1.5, 
                    zorder=5, label='Z* (Punto Ideal)') # zorder=5 asegura que quede arriba

    # Configuración del gráfico
    plt.title(f"Frente de Pareto y Población Final (Gen: {current_gen})")
    plt.xlabel('Objetivo 1: Dice Loss (Minimizar)')
    plt.ylabel('Objetivo 2: Norm Params (Minimizar)')
    
    # Fijar los ejes (usando los límites definidos)
    plt.xlim(0.02, X_MAX_LIMIT)
    plt.ylim(-0.001, Y_MAX_LIMIT)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Guardar el gráfico
    plt.savefig(OUTPUT_FILE)
    plt.close()

    print(f"\n✅ Gráfico clasificado guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_final_pareto_front()
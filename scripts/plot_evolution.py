import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- CONFIGURACIÓN ---
LOG_FILE = "moead_dl_log.json"
OUTPUT_FOLDER = "evolution_plots"

def create_evolution_sequence():
    """Lee el log JSON y genera una imagen del Frente de Pareto para cada generación."""
    print(f"--> Leyendo log histórico: {LOG_FILE}")
    
    # 1. Cargar el historial completo
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            # Cargar el diccionario principal
            data = json.load(f)
            
            # Asignar la lista de generaciones a 'history' (el cambio clave)
            history = data.get('metadata', {}).get('generations_data', [])
            
    except FileNotFoundError:
        print(f"ERROR: Archivo {LOG_FILE} no encontrado. Asegúrate de la ruta.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: No se pudo decodificar el archivo JSON.")
        return

    # Verificación de que history es una lista válida
    if not isinstance(history, list):
        print("ERROR: La estructura del log no contiene 'metadata/generations_data' como lista.")
        return
    # Crear la carpeta de salida
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    
    # 2. Determinar límites globales para que el gráfico no salte
    # Usamos los datos de la última generación para establecer límites estables
    all_loss = [s['objectives'][0] for gen in history for s in gen['solutions'] if s.get('objectives')]
    all_params = [s['objectives'][1] for gen in history for s in gen['solutions'] if s.get('objectives')]

    if not all_loss or not all_params:
        print("ERROR: No se encontraron soluciones válidas en el log.")
        return

    # Definir límites del gráfico (para que los ejes no cambien en cada frame)
    X_MAX = np.clip(max(all_loss) * 1.1, 0, 1.05)
    Y_MAX = np.clip(max(all_params) * 1.1, 0, 1.05)
    
    print(f"--> Generando {len(history)} imágenes del Frente de Pareto...")

    # 3. Iterar por cada generación
    for gen_data in history:
        gen_num = gen_data['generation']
        solutions = gen_data['solutions']
        
        # Extraer solo las soluciones válidas y sus objetivos
        valid_solutions = [s for s in solutions if s.get('objectives')]
        if not valid_solutions: continue

        loss_values = [s['objectives'][0] for s in valid_solutions]
        param_values = [s['objectives'][1] for s in valid_solutions]

        plt.figure(figsize=(8, 6))
        
        # Graficar todos los individuos de esta generación
        plt.scatter(loss_values, param_values, 
                    s=20, alpha=0.7, color='blue', edgecolors='black', linewidths=0.5)

        # Configuración del gráfico
        plt.title(f"Gen {gen_num} / {len(history)-1}: Evolución del Frente de Pareto")
        plt.xlabel('Dice Loss (Minimizar)')
        plt.ylabel('Norm Params (Minimizar)')
        
        # Fijar los ejes para la animación
        plt.xlim(0, X_MAX)
        plt.ylim(-0.002, Y_MAX)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Guardar el frame
        # Usamos un formato de 3 dígitos (ej: 001, 002) para que el software de GIF ordene bien
        frame_name = Path(OUTPUT_FOLDER) / f"frame_{gen_num:03d}.png"
        plt.savefig(frame_name)
        plt.close()

    print("\n✅ Secuencia de frames generada exitosamente.")
    print(f"   Archivos guardados en la carpeta: {OUTPUT_FOLDER}")
    print("   Usa un creador de GIFs para animar la secuencia.")

if __name__ == "__main__":
    create_evolution_sequence()
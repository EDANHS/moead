"""
Capa de Visualización y Presentación Analítica (Thesis Plotting Engine).
Implementa un patrón de Auto-Descubrimiento (Auto-Discovery) combinado con 
Traducción Semántica Condicional para mapear identificadores de máquina 
a etiquetas académicas. Renderiza dinámicamente curvas de convergencia MLOps.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. TRADUCCIÓN SEMÁNTICA (LEXICOGRAPHICAL MAPPING)
# ==========================================
def format_algorithm_name(algo_id: str) -> str:
    """
    Motor de traducción condicional.
    Mapea el identificador crudo del Data Mart (ej. 'uniform_crossover') a una
    etiqueta académica de alto nivel, formateada para el documento de tesis.
    """
    if algo_id == "uniform_crossover":
        return "Cruce Uniforme (UX+Tcheb)"
    elif algo_id == "differential_evolution":
        return "Evolución Diferencial (DE+PBI)"
    else:
        # Fallback de seguridad para algoritmos futuros no catalogados
        return algo_id.replace('_', ' ').title()

# ==========================================
# 2. MOTOR DE INGESTIÓN (AUTO-DISCOVERY)
# ==========================================
def load_data_marts(data_marts_dir: Path) -> dict:
    """
    Escanea el directorio objetivo, hidrata la memoria con los Data Marts
    y estructura un diccionario consolidado extrayendo el ID interno de cada JSON.
    """
    print(f"--> [INICIO] Ejecutando Auto-Descubrimiento de Data Marts en: {data_marts_dir}")
    
    consolidated_data = {}
    json_files = list(data_marts_dir.glob("*.json"))
    
    if not json_files:
        print("[ERROR CRÍTICO] No se encontraron archivos JSON en el directorio especificado.")
        return consolidated_data
        
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
                
            # Extracción del ID interno y traducción semántica automática
            raw_id = payload.get("algorithm_id", file_path.stem)
            academic_label = format_algorithm_name(raw_id)
            
            consolidated_data[academic_label] = payload
            print(f"  -> [Cargado] Telemetría mapeada: {academic_label} ({payload.get('generations_processed', 0)} generaciones)")
            
        except Exception as e:
            print(f"  -> [ERROR] Fallo al cargar {file_path.name}: {e}")
            
    return consolidated_data

# ==========================================
# 3. CAPA DE RENDERIZADO VISUAL (PLOTTER)
# ==========================================
def plot_comparative_metric(data: dict, metric_key: str, title: str, ylabel: str, output_path: Path):
    """
    Motor de renderizado agnóstico. 
    Itera sobre la data consolidada, asignando paletas de color consistentes 
    y alineando tensores asimétricos para una validación cruzada perfecta.
    """
    plt.figure(figsize=(10, 6))
    
    # Paleta corporativa de alta legibilidad y jerarquía visual
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Ordenamiento alfabético para forzar consistencia de colores sin importar cómo el OS lea los archivos
    sorted_algorithms = sorted(data.items())
    
    for idx, (academic_label, payload) in enumerate(sorted_algorithms):
        y_values = payload.get(metric_key, [])
        if not y_values:
            print(f"  -> [WARN] Métrica '{metric_key}' no encontrada en {academic_label}. Omitiendo curva.")
            continue
            
        # Generación dinámica del Eje X permitiendo asimetría temporal (Early Truncation)
        x_values = np.arange(len(y_values))
        
        # Asignación estética modular
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Trazado de la curva y sombreado de área (Impacto visual MLOps)
        plt.plot(x_values, y_values, marker=marker, markersize=7, linewidth=2.5, color=color, label=academic_label, alpha=0.9)
        plt.fill_between(x_values, y_values, color=color, alpha=0.08)

    # Parametrización del lienzo y tipografías
    plt.xlabel('Progreso Evolutivo (Generaciones)', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, pad=15)
    
    # Forzar ticks enteros en el eje X para las generaciones
    ax = plt.gca()
    ax.xaxis.get_major_locator().set_params(integer=True)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11, shadow=True, framealpha=0.95)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"--> [RENDERIZADO EXITOSO] Gráfico exportado a: {output_path.name}")

# ==========================================
# 4. ORQUESTADOR DE PRESENTACIÓN
# ==========================================
if __name__ == '__main__':
    # ---------------------------------------------------------
    # CONFIGURACIÓN DE RUTAS (ZERO-CONFIG PARA ALGORITMOS)
    # ---------------------------------------------------------
    # Directorio donde tu script ETL guardó los Data Marts
    DATA_MARTS_DIR = Path("data_marts")
    
    # Directorio donde se guardarán las imágenes finales de tu tesis
    OUTPUT_DIR = Path("plots")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # EJECUCIÓN DEL PIPELINE VISUAL
    # ---------------------------------------------------------
    # La hidratación en memoria ahora es un proceso completamente automatizado
    consolidated_payloads = load_data_marts(DATA_MARTS_DIR)
    
    if consolidated_payloads:
        print("\n--> [FASE 1] Renderizando Perfilado Físico (Costo Temporal)...")
        plot_comparative_metric(
            data=consolidated_payloads,
            metric_key="generation_times_minutes",
            title="Estudio de Aceleración Computacional por Generación Aislada\n(Evidencia de Eficiencia de Memoria Caché)",
            ylabel="Latencia Computacional (Minutos / Generación)",
            output_path=OUTPUT_DIR / "plot_01_generational_times.png"
        )
        
        print("\n--> [FASE 2] Renderizando Convergencia de Eficacia Clínica...")
        plot_comparative_metric(
            data=consolidated_payloads,
            metric_key="average_dice_loss",
            title="Convergencia de Precisión de Segmentación Clínica (NAS)\nEvolución del Error Promedio Poblacional",
            ylabel="Promedio de Dice Loss (Minimizar)",
            output_path=OUTPUT_DIR / "plot_02_average_dice.png"
        )
        
        print("\n--> [FASE 3] Renderizando Convergencia de Eficiencia Topológica...")
        plot_comparative_metric(
            data=consolidated_payloads,
            metric_key="average_norm_params",
            title="Evolución de la Compresión Topológica Arquitectónica\nReducción Promedio del Costo Paramétrico",
            ylabel="Promedio de Parámetros Normalizados (Minimizar)",
            output_path=OUTPUT_DIR / "plot_03_average_params.png"
        )
        
        print(f"\n--> [CLÍMAX ALCANZADO] Todos los gráficos de tu tesis están listos en el directorio: {OUTPUT_DIR}")
        print("¡Corre a insertarlos en tu documento y asegúrate el título de Ingeniero!")
"""
Herramienta de Extracción y Estandarización de Telemetría (ETL Log Parser).
Escanea la salida estándar, aísla los bloques generacionales (incluyendo la Gen 0)
y exporta una matriz dimensionalmente simétrica (N_Generaciones x 50) a un JSON
persistente para permitir el análisis comparativo post-hoc (Apples-to-Apples).
"""

import re
import json
from pathlib import Path
from collections import defaultdict

def extract_and_standardize_telemetry(log_filepath: str, n_pop: int = 50) -> dict:
    """
    Núcleo ETL (Extract, Transform, Load) basado en Máquina de Estados.
    Extrae los tiempos físicos y aplica 'Zero-Padding' deductivo para representar
    matemáticamente los Cache Hits, garantizando tensores de tamaño fijo.
    """
    print(f"--> [INICIO] Extrayendo telemetría cruda desde: {log_filepath}")
    
    # Memoria transaccional temporal
    raw_times_by_gen = defaultdict(list)
    
    # Máquina de Estados: Patrones de Transición
    init_pattern = re.compile(r"Generando/Completando población inicial")
    gen_pattern = re.compile(r"---\s*Generaci[óo]n\s+(\d+)/\d+\s*---")
    time_pattern = re.compile(r"Tiempo:\s*([\d\.]+)s")
    
    current_gen = None
    
    # 1. FASE DE EXTRACCIÓN (EXTRACT)
    try:
        with open(log_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Detección del Arranque en Frío (Cold Start)
                if init_pattern.search(line):
                    current_gen = 0
                    continue
                
                # Transición a nueva generación
                gen_match = gen_pattern.search(line)
                if gen_match:
                    current_gen = int(gen_match.group(1))
                    continue
                
                # Recolección del costo físico
                time_match = time_pattern.search(line)
                if time_match and current_gen is not None:
                    time_in_seconds = float(time_match.group(1))
                    raw_times_by_gen[current_gen].append(time_in_seconds)
                    
    except FileNotFoundError:
        print(f"[ERROR CRÍTICO] El archivo log '{log_filepath}' no existe.")
        return {}

    # 2. FASE DE TRANSFORMACIÓN (TRANSFORM & ZERO-PADDING)
    print("--> [TRANSFORMACIÓN] Estandarizando dimensionalidad de la matriz (Padding de Cache Hits)...")
    
    structured_telemetry = {
        "metadata": {
            "source_log": str(log_filepath),
            "subproblems_per_gen": n_pop,
            "total_generations_captured": len(raw_times_by_gen)
        },
        "generations": {}
    }
    
    for gen in sorted(raw_times_by_gen.keys()):
        physical_times = raw_times_by_gen[gen]
        cache_misses = len(physical_times)
        
        # Deducción geométrica: Si la población es 50 y evaluamos 12, hubo 38 Cache Hits.
        cache_hits = n_pop - cache_misses
        
        # Relleno de ceros (El costo de un Cache Hit es 0.0 segundos)
        padded_times = physical_times + [0.0] * cache_hits
        
        # Cálculo de agregados para facilitar el ploteo futuro
        total_time_seconds = sum(physical_times)
        total_time_minutes = total_time_seconds / 60.0
        
        structured_telemetry["generations"][str(gen)] = {
            "cache_misses": cache_misses,
            "cache_hits": cache_hits,
            "total_time_minutes": total_time_minutes,
            # La matriz dimensional perfecta de 50 elementos:
            "times_array": padded_times 
        }
        
    return structured_telemetry

def export_to_json(data: dict, output_filepath: str):
    """
    Fase de Carga (LOAD). Persiste la estructura analítica en el disco.
    """
    if not data:
        print("--> [ABORTADO] No hay datos válidos para exportar.")
        return
        
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"--> [ÉXITO] Matriz de telemetría exportada impecablemente a: {output_filepath}")
    except Exception as e:
        print(f"--> [ERROR] Fallo al escribir el JSON: {e}")

if __name__ == '__main__':
    # Configuración de Rutas y Constantes
    N_POPULATION = 50 # Tamaño fijo del vecindario/subproblemas en tu MOEAD
    
    # Modifica esto con el nombre exacto de tu archivo log
    LOG_FILE_PATH = "ejecucion_moead.log" 
    
    # Archivo JSON de salida (Tu nuevo almacén de datos temporal)
    OUTPUT_JSON = "telemetry_uniform_crossover.json"
    
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    
    # Orquestación del Pipeline ETL
    telemetry_data = extract_and_standardize_telemetry(LOG_FILE_PATH, N_POPULATION)
    export_to_json(telemetry_data, OUTPUT_JSON)
"""
Módulo de Extracción y Transformación de Telemetría (ETL Unitario).
Este script consolida las métricas de evaluación de hardware (tiempos) y 
rendimiento fenotípico (Dice/Params) de un ÚNICO algoritmo evolutivo.
Implementa 'Truncamiento Analítico Simétrico' para garantizar equidad 
comparativa en experimentos interrumpidos asimétricamente.
"""

import re
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ==========================================
# 1. EXTRACCIÓN DE TELEMETRÍA (LOG PARSER)
# ==========================================
def extract_generational_times(log_filepaths: list[str], target_gen: int) -> list[float]:
    """
    Concatena secuencialmente múltiples archivos de log y extrae el tiempo 
    aislado consumido por CADA generación, truncando el horizonte temporal
    hasta 'target_gen' (incluyendo la Gen 0).
    """
    raw_times_by_gen = defaultdict(list)
    
    init_pattern = re.compile(r"Generando/Completando población inicial")
    gen_pattern = re.compile(r"---\s*Generaci[óo]n\s+(\d+)/\d+\s*---")
    miss_pattern = re.compile(r"\[CACHE MISS\]")
    hit_pattern = re.compile(r"\[CACHE HIT\]")
    time_pattern = re.compile(r"Tiempo:\s*([\d\.]+)s")
    
    current_gen = None
    pending_miss = False
    
    for filepath in log_filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if init_pattern.search(line):
                        if pending_miss and current_gen is not None:
                            raw_times_by_gen[current_gen].append(0.0)
                        current_gen = 0
                        pending_miss = False
                        continue
                    
                    gen_match = gen_pattern.search(line)
                    if gen_match:
                        if pending_miss and current_gen is not None:
                            raw_times_by_gen[current_gen].append(0.0)
                        current_gen = int(gen_match.group(1))
                        pending_miss = False
                        continue
                    
                    if miss_pattern.search(line) and current_gen is not None:
                        if pending_miss:
                            raw_times_by_gen[current_gen].append(0.0)
                        pending_miss = True
                        
                    elif hit_pattern.search(line) and current_gen is not None:
                        if pending_miss:
                            raw_times_by_gen[current_gen].append(0.0)
                            pending_miss = False
                        raw_times_by_gen[current_gen].append(0.0)
                        
                    time_match = time_pattern.search(line)
                    if time_match and current_gen is not None and pending_miss:
                        raw_times_by_gen[current_gen].append(float(time_match.group(1)))
                        pending_miss = False
                        
        except FileNotFoundError:
            print(f"  [ADVERTENCIA] Log no encontrado: {filepath}. Omitiendo archivo en el pipeline.")

    if not raw_times_by_gen:
        return []

    # Aplicación del Truncamiento Matemático del Horizonte
    max_gen_available = max(raw_times_by_gen.keys())
    
    # Se asegura de no pedir más generaciones de las que existen realmente
    horizon_limit = min(max_gen_available, target_gen)
    
    generation_times = []
    
    for gen in range(horizon_limit + 1): # +1 para asegurar que se incluye el límite
        gen_time_seconds = sum(raw_times_by_gen.get(gen, [0.0]))
        generation_times.append(gen_time_seconds / 60.0) # Normalización a minutos
        
    return generation_times

# ==========================================
# 2. VALIDACIÓN CRUZADA: HISTORY VS CACHE
# ==========================================
def create_robust_hash(config_dict: dict) -> tuple:
    """Crea una firma inmutable para el cruce relacional."""
    if not isinstance(config_dict, dict):
        return tuple()
    hashable_items = []
    for k, v in sorted(config_dict.items()):
        if isinstance(v, list):
            v = tuple(v)
        hashable_items.append((k, v))
    return tuple(hashable_items)

def extract_generational_metrics(history_path: str, cache_path: str, target_gen: int) -> tuple[list[float], list[float]]:
    """
    Ejecuta un 'Data Join' empírico, aislando topológicamente el análisis
    hasta la generación definida por 'target_gen'.
    """
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            hist_data = json.load(f)
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
    except FileNotFoundError as e:
        print(f"  [ADVERTENCIA] Deficiencia de archivos métricos: {e}")
        return [], []

    robust_cache = {}
    for raw_key_str, metrics in cache_data.items():
        try:
            config_dict = json.loads(raw_key_str)
            robust_key = create_robust_hash(config_dict)
            robust_cache[robust_key] = metrics.get('objectives', [1.0, 1.0])
        except Exception:
            continue

    pop_history = hist_data.get('history', {}).get('population_history', [])
    
    # Truncamiento Seguro de la Memoria Temporal
    max_history_gen = len(pop_history) - 1
    horizon_limit = min(max_history_gen, target_gen)
    
    # Slice matemático: Incluye desde Gen 0 hasta el límite de forma precisa
    pop_history_truncated = pop_history[:horizon_limit + 1]

    avg_dices = []
    avg_params = []

    for gen_snapshot in pop_history_truncated:
        gen_dices = []
        gen_params = []
        
        for sol in gen_snapshot:
            config = sol.get('model_config')
            fallback_obj = sol.get('objectives', [1.0, 1.0])
            
            if config:
                robust_key = create_robust_hash(config)
                if robust_key in robust_cache:
                    obj = robust_cache[robust_key]
                else:
                    obj = fallback_obj
            else:
                obj = fallback_obj
                
            gen_dices.append(obj[0])
            gen_params.append(obj[1])
            
        avg_dices.append(float(np.mean(gen_dices)) if gen_dices else 1.0)
        avg_params.append(float(np.mean(gen_params)) if gen_params else 1.0)

    return avg_dices, avg_params

# ==========================================
# 3. ORQUESTADOR UNITARIO DE CONSOLIDACIÓN
# ==========================================
if __name__ == '__main__':
    # ---------------------------------------------------------
    # PANEL DE CONFIGURACIÓN
    # ---------------------------------------------------------
    
    # [NUEVO] Horizonte Temporal de Truncamiento (Ej. 8 significa de Gen 0 a Gen 8)
    TARGET_GEN = 25
    
    ALGORITHM_ID = "uniform_crossover" 
    
    LOG_FILES = [
        "archivos_cache/ejecucion_moead.log"
    ]
    HISTORY_FILE = "archivos_cache/uniform_moead_dl_checkpoint_ctv.pkl.json"
    CACHE_FILE = "archivos_cache/nas_evaluation_cache_ctv.json"
    
    OUTPUT_DIR = Path("data_marts")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / f"metrics_{ALGORITHM_ID}.json"
    
    # ---------------------------------------------------------
    # PIPELINE DE EJECUCIÓN
    # ---------------------------------------------------------
    print(f"--> [INICIO] Arrancando ETL Unitario para: {ALGORITHM_ID.upper()} (Truncado en Gen {TARGET_GEN})")
    
    print("  -> FASE 1: Extrayendo y consolidando la Cronometría Acumulada de Keras...")
    times_array = extract_generational_times(LOG_FILES, TARGET_GEN)
    
    print("  -> FASE 2: Auditando topología y cruzando Métricas Fenotípicas (Dice/Params)...")
    dices_array, params_array = extract_generational_metrics(HISTORY_FILE, CACHE_FILE, TARGET_GEN)
    
    print(f"  -> FASE 3: Generando Data Mart (JSON) en {OUTPUT_FILE}...")
    
    # Garantizamos que la llave exportada haga match exacto con el Ploter Final
    consolidated_payload = {
        "algorithm_id": ALGORITHM_ID,
        "generations_processed": len(times_array),
        "generation_times_minutes": times_array, 
        "average_dice_loss": dices_array,
        "average_norm_params": params_array
    }
    
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(consolidated_payload, f, indent=4)
        print(f"--> [ÉXITO] Pipeline completado. Tensor simétrico de {len(times_array)} instantes exportado correctamente.")
    except Exception as e:
        print(f"--> [ERROR CRÍTICO] Fallo al persistir el Data Mart: {e}")
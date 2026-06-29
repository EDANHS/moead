import json
import os
from pathlib import Path

def merge_nas_caches(file_a_path: str, file_b_path: str, output_path: str, strategy: str = 'best'):
    """
    Consolida dos bases de datos de cache de MOEA/D en un único archivo maestro.
    Resuelve colisiones de topologías priorizando el rendimiento elitista.
    """
    print("[*] Iniciando proceso de consolidación de bases de datos históricas...")
    
    # 1. Carga de archivos fuente
    if not os.path.exists(file_a_path) or not os.path.exists(file_b_path):
        print("[ERROR] Uno o ambos archivos de caché especificados no existen.")
        return

    with open(file_a_path, 'r', encoding='utf-8') as f:
        cache_a = json.load(f)
    with open(file_b_path, 'r', encoding='utf-8') as f:
        cache_b = json.load(f)

    print(f"    -> Registros en Caché A: {len(cache_a)}")
    print(f"    -> Registros en Caché B: {len(cache_b)}")

    # 2. Inicialización del repositorio maestro
    master_cache = dict(cache_a) # Copiamos el primer bloque completo
    duplicate_count = 0
    overwritten_count = 0

    # 3. Bucle de resolución y combinación analítica
    for config_str, data_b in cache_b.items():
        if config_str in master_cache:
            duplicate_count += 1
            data_a = master_cache[config_str]
            
            # Suponemos que objectives[0] es la función de pérdida (Dice Loss) a minimizar
            obj_a = data_a["objectives"][0]
            obj_b = data_b["objectives"][0]
            
            if strategy == 'best':
                # Si el nuevo dato (B) es mejor (menor pérdida) que el existente (A)
                if obj_b < obj_a:
                    master_cache[config_str] = data_b
                    overwritten_count += 1
            elif strategy == 'worst':
                # Si se prefiere un enfoque conservador de límite inferior
                if obj_b > obj_a:
                    master_cache[config_str] = data_b
                    overwritten_count += 1
        else:
            # Si la topología es inédita, se indexa directamente
            master_cache[config_str] = data_b

    # 4. Persistencia e integridad de los datos unificados
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(master_cache, f, indent=2, sort_keys=True)

    # 5. Reporte métrico de la unificación
    print("\n[+] Proceso de fusión finalizado con éxito.")
    print(f"    -> Total de arquitecturas únicas en base maestra: {len(master_cache)}")
    print(f"    -> Arquitecturas redundantes detectadas: {duplicate_count}")
    if strategy == 'best':
        print(f"    -> Registros actualizados con una evaluación superior: {overwritten_count}")
    print(f"    -> Archivo maestro guardado en: {output_path}")

if __name__ == "__main__":
    # Define las rutas de tus exploraciones previas
    CACHE_EXPLORACION_1 = "archivos_cache/nas_evaluation_cache_ctv.json" 
    CACHE_EXPLORACION_2 = "archivos_cache/nas_evaluation_cache_de_ctv.json" 
    CACHE_MAESTRO = "archivos_cache/nas_evaluation_cache_master.json" 
    
    # Ejecución del pipeline de curaduría
    merge_nas_caches(
        file_a_path=CACHE_EXPLORACION_1,
        file_b_path=CACHE_EXPLORACION_2,
        output_path=CACHE_MAESTRO,
        strategy='best' # Elitismo estructural
    )
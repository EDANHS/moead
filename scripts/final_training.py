import os
import json
import gc
import time
import traceback
import numpy as np
from pathlib import Path
from multiprocessing import Process, Queue
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN DE ENTORNO (CRÍTICO PARA ESTABILIDAD EN RTX 5080/SERIES) ---
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def to_scientific_notation(num: int) -> str:
    """Convierte un entero a formato científico N x 10^X legible para tesis."""
    fmt = f"{num:.2e}"  # Ejemplo: '1.54e+06'
    base, exponent = fmt.split('e')
    exponent = int(exponent)  # Elimina signos + y ceros a la izquierda
    return f"{base} x 10^{exponent}"

def isolated_evaluation_worker(queue, compromise_key, model_config, X_path, Y_path, epochs=100, patience=5):
    """
    PROCESO HIJO TOTALMENTE AISLADO.
    TensorFlow e hilos de CUDA nacen y mueren exclusivamente aquí.
    """
    # Parches elásticos para control estricto de VRAM
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_CACHE_MAXSIZE'] = '4294967296'
    
    raw_params = 0
    try:
        # 1. Carga determinista y partición de datos dentro del subproceso
        # Al usar el mismo random_state, cada worker operará exactamente sobre los mismos conjuntos.
        X = np.load(X_path).astype(np.float32)
        Y = np.load(Y_path).astype(np.float32)
        input_shape = X.shape[1:]

        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y, test_size=0.30, random_state=42, shuffle=True
        )
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=0.50, random_state=42, shuffle=True
        )
        del X, Y, X_temp, Y_temp  # Liberación inmediata de RAM
        
        # 2. Lazy Loading de módulos pesados de TensorFlow
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras import mixed_precision
        from moead.models import build_unet
        from moead.utils.tf_metrics import dice_coefficient, dice_loss

        # Activación de precisión mixta para acelerar el entrenamiento
        mixed_precision.set_global_policy("mixed_float16")

        # Configuración dinámica de VRAM
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # 3. Construcción del modelo específico
        print(f"\n[Worker - {compromise_key}] Construyendo Arquitectura...")
        model = build_unet(input_shape, **model_config)
        raw_params = model.count_params()

        # 4. Compilación con Gradiente Acumulado (Simula lotes grandes sin saturar VRAM)
        optimizador = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            gradient_accumulation_steps=2
        )
        model.compile(
            optimizer=optimizador,
            loss=dice_loss,
            metrics=[dice_coefficient],
            jit_compile=False
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
        ]

        # 5. Entrenamiento Intensivo
        print(f"[Worker - {compromise_key}] Iniciando entrenamiento de {epochs} épocas...")
        model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            batch_size=8,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # 6. Evaluación en el conjunto de TEST (Datos completamente inéditos)
        print(f"[Worker - {compromise_key}] Evaluando en Set de Test...")
        eval_results = model.evaluate(X_test, Y_test, batch_size=8, verbose=0)
        final_test_dice = float(eval_results[1])  # Índice 1 corresponde a dice_coefficient

        # Enviar resultados exitosos al proceso padre
        queue.put({
            "compromise": compromise_key,
            "success": True,
            "params": raw_params,
            "test_dice": final_test_dice,
            "error": None
        })

    except Exception as e:
        queue.put({
            "compromise": compromise_key,
            "success": False,
            "params": raw_params if 'raw_params' in locals() else 0,
            "test_dice": 0.0,
            "error": traceback.format_exc()
        })
        
    finally:
        # Purga agresiva previa a la muerte del proceso
        tf.keras.backend.clear_session()
        local_vars = ['model', 'X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test']
        for var in local_vars:
            if var in locals():
                del locals()[var]
        gc.collect()


def main():
    # --- CONFIGURACIÓN DE RUTAS ---
    COMPROMISES_JSON = "pareto_compromises_gen_8_ux.json"  # Tu JSON generado en el paso anterior
    X_DATA_PATH = Path("data/X_train_ctv_5k.npy")
    Y_DATA_PATH = Path("data/Y_train_ctv_5k.npy")
    CONSOLIDATED_OUTPUT = Path("resultados_finales/consolidado_entrenamiento_pareto.json")
    
    CONSOLIDATED_OUTPUT.parent.mkdir(exist_ok=True)

    if not os.path.exists(COMPROMISES_JSON):
        print(f"[ERROR] No se encuentra el archivo de compromisos: {COMPROMISES_JSON}")
        return

    with open(COMPROMISES_JSON, 'r', encoding='utf-8') as f:
        compromise_data = json.load(f)

    compromises = compromise_data.get("compromises", {})
    final_results = {
        "meta": {
            "source_compromises": COMPROMISES_JSON,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_split": "70% Train, 15% Val, 15% Test"
        },
        "models": {}
    }

    # Iterar secuencialmente sobre cada perfil para garantizar el aislamiento total de memoria
    for key, info in compromises.items():
        print(f"\n" + "="*70)
        print(f" Preparation de lanzamiento: Perfil [{key}]")
        print(f" Descripción: {info['descripcion']}")
        print("="*70)

        # Extraer la configuración del modelo desde el JSON consolidado previo
        # Ajustamos dinámicamente si la clave cuelga directamente o de 'model_config'
        model_config = info["datos_red"].get("model_config") or info["datos_red"].get("architecture_config")
        
        if not model_config:
            print(f"[ADVERTENCIA] No se encontró configuración de modelo para {key}. Saltando...")
            continue

        # Crear una cola de comunicación dedicada para este subproceso
        queue = Queue()

        # Instanciar el proceso hijo aislado
        p = Process(
            target=isolated_evaluation_worker, 
            args=(queue, key, model_config, X_DATA_PATH, Y_DATA_PATH), # Ajustado abajo a cadenas de texto de ruta
            kwargs={"epochs": 100, "patience": 5}
        )
        
        # Corregir paso de rutas como String por compatibilidad de serialización en multiproceso
        p = Process(
            target=isolated_evaluation_worker,
            args=(queue, key, model_config, str(X_DATA_PATH), str(Y_DATA_PATH)),
            kwargs={"epochs": 100, "patience": 5}
        )

        p.start()
        
        # Esperar a que el proceso termine por completo antes de continuar con la siguiente arquitectura
        p.join()

        # Recoger los resultados enviados a través de la cola
        if not queue.empty():
            res = queue.get()
            if res["success"]:
                print(f"--> [ÉXITO - {key}] Test Dice Score obtenido: {res['test_dice']:.4f}")
                
                # Almacenamiento limpio según los requisitos de tu tesis
                final_results["models"][key] = {
                    "descripcion": info['descripcion'],
                    "parametros_notacion_cientifica": to_scientific_notation(res["params"]),
                    "parametros_raw": res["params"],
                    "test_dice_score": round(res["test_dice"], 4)
                }
            else:
                print(f"--> [FALLO - {key}] El proceso terminó con errores.")
                print(res["error"])
                final_results["models"][key] = {
                    "success": False,
                    "error": "RuntimeError durante el aislamiento"
                }
        else:
            print(f"--> [CRÍTICO - {key}] El proceso hijo murió abruptamente sin devolver datos.")

    # Guardar el JSON consolidado final
    with open(CONSOLIDATED_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"\n✨ [PROCESO GLOBAL COMPLETADO] Resultados unificados en: {CONSOLIDATED_OUTPUT}")

if __name__ == '__main__':
    main()
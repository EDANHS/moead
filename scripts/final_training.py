import os
import sys
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
    fmt = f"{num:.2e}"
    base, exponent = fmt.split('e')
    exponent = int(exponent)
    return f"{base} x 10^{exponent}"

def isolated_evaluation_worker(queue, compromise_key, model_config, X_path, Y_path, save_dir, epochs=100, patience=5):
    """
    PROCESO HIJO TOTALMENTE AISLADO.
    TensorFlow e hilos de CUDA nacen y mueren exclusivamente aquí.
    """
    # Parches estelares para control estricto de VRAM
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_CACHE_MAXSIZE'] = '4294967296'
    
    raw_params = 0
    try:
        # 1. Carga determinista y partición de datos
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
        
        # 2. Lazy Loading de módulos
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras import mixed_precision
        from moead.models import build_unet
        from moead.utils.tf_metrics import dice_coefficient, dice_loss
        
        # Lazy Loading seguro de Matplotlib para subprocesos
        import matplotlib
        matplotlib.use('Agg')  # Backend 'Agg' previene errores de GUI en multiprocesamiento
        import matplotlib.pyplot as plt

        mixed_precision.set_global_policy("mixed_float16")

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # 3. Construcción del modelo
        print(f"\n[Worker - {compromise_key}] Construyendo Arquitectura...")
        model = build_unet(input_shape, **model_config)
        raw_params = model.count_params()

        # 4. Compilación
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

        # 5. Entrenamiento Intensivo y captura del historial
        print(f"[Worker - {compromise_key}] Iniciando entrenamiento de {epochs} épocas...")
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            batch_size=8,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # 6. Evaluación en el conjunto de TEST
        print(f"[Worker - {compromise_key}] Evaluando en Set de Test...")
        eval_results = model.evaluate(X_test, Y_test, batch_size=8, verbose=0)
        final_test_dice = float(eval_results[1])

        # 7. GUARDAR EL MODELO ENTRENADO (.keras y .h5)
        model_save_path_keras = os.path.join(save_dir, f"{compromise_key}.keras")
        model_save_path_h5 = os.path.join(save_dir, f"{compromise_key}.h5")
        model.save(model_save_path_keras)
        model.save(model_save_path_h5)
        
        # 8. GENERAR Y GUARDAR GRÁFICA DE PÉRDIDA
        plot_save_path = os.path.join(save_dir, f"{compromise_key}_loss_plot.png")
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Entrenamiento (Train Loss)', color='blue', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validación (Val Loss)', color='red', linestyle='--', linewidth=2)
        plt.title(f'Curva de Pérdida (Dice Loss) - {compromise_key}', fontsize=14)
        plt.xlabel('Épocas', fontsize=12)
        plt.ylabel('Pérdida', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plot_save_path, dpi=300)
        plt.close() # Cerrar figura para liberar memoria RAM
        
        print(f"[Worker - {compromise_key}] Artefactos guardados exitosamente en la carpeta de destino.")

        # Enviar resultados al proceso padre
        queue.put({
            "compromise": compromise_key,
            "success": True,
            "params": raw_params,
            "test_dice": final_test_dice,
            "plot_path": f"{compromise_key}_loss_plot.png",
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
        if 'tf' in locals():
            tf.keras.backend.clear_session()
        local_vars = ['model', 'X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test', 'history']
        for var in local_vars:
            if var in locals():
                del locals()[var]
        gc.collect()


def main():
    if len(sys.argv) < 2:
        print("[ERROR] Debes proporcionar el nombre de la etiqueta como argumento.")
        print("Uso correcto: python script.py <nombre_de_la_etiqueta>")
        sys.exit(1)
        
    etiqueta_carpeta = sys.argv[1]

    COMPROMISES_JSON = "pareto_compromises_gen_25_uniform.json"
    X_DATA_PATH = Path("data/X_train_ctv_5k.npy")
    Y_DATA_PATH = Path("data/Y_train_ctv_5k.npy")
    
    BASE_OUTPUT_DIR = Path("resultados_finales") / etiqueta_carpeta
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLIDATED_OUTPUT = BASE_OUTPUT_DIR / "consolidado_entrenamiento_pareto.json"

    if not os.path.exists(COMPROMISES_JSON):
        print(f"[ERROR] No se encuentra el archivo: {COMPROMISES_JSON}")
        return

    with open(COMPROMISES_JSON, 'r', encoding='utf-8') as f:
        compromise_data = json.load(f)

    compromises = compromise_data.get("compromises", {})
    final_results = {
        "meta": {
            "source_compromises": COMPROMISES_JSON,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_split": "70% Train, 15% Val, 15% Test",
            "etiqueta_directorio": etiqueta_carpeta
        },
        "models": {}
    }

    for key, info in compromises.items():
        print(f"\n" + "="*70)
        print(f" Preparación de lanzamiento: Perfil [{key}]")
        print(f" Descripción: {info['descripcion']}")
        print("="*70)

        model_config = info["datos_red"].get("model_config") or info["datos_red"].get("architecture_config")
        if not model_config:
            continue

        queue = Queue()
        p = Process(
            target=isolated_evaluation_worker,
            args=(queue, key, model_config, str(X_DATA_PATH), str(Y_DATA_PATH), str(BASE_OUTPUT_DIR)),
            kwargs={"epochs": 100, "patience": 5}
        )

        p.start()
        p.join()

        if not queue.empty():
            res = queue.get(timeout=7200)
            if res["success"]:
                print(f"--> [ÉXITO - {key}] Test Dice Score obtenido: {res['test_dice']:.4f}")
                
                final_results["models"][key] = {
                    "descripcion": info['descripcion'],
                    "parametros_notacion_cientifica": to_scientific_notation(res["params"]),
                    "parametros_raw": res["params"],
                    "test_dice_score": round(res["test_dice"], 4),
                    "saved_formats": [f"{key}.keras", f"{key}.h5", res["plot_path"]]
                }
            else:
                print(f"--> [FALLO - {key}] El proceso terminó con errores.")
                print(res["error"])
                final_results["models"][key] = {
                    "success": False,
                    "error": "RuntimeError durante el aislamiento"
                }
        else:
            print(f"--> [CRÍTICO - {key}] El proceso hijo murió.")

    with open(CONSOLIDATED_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"\n✨ [PROCESO GLOBAL COMPLETADO] Resultados unificados en: {CONSOLIDATED_OUTPUT}")

if __name__ == '__main__':
    main()
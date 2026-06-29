import json
import gc
import time
from pathlib import Path
import tensorflow as tf
import numpy as np

# Ajusta esta importación a la ruta real de tu constructor en el proyecto
from moead.models import build_unet 

def compute_zero_cost_proxies(model, input_shape=(256, 256, 1)):
    """
    Extracción simultánea del ensamble ortogonal de Zero-Cost Proxies.
    Retorna: (synflow_score, snip_score, jacobian_score)
    """
    # =====================================================================
    # FASE 1: PROXIES DE GRADIENTE (Batch = 1, Máxima Velocidad)
    # =====================================================================
    dummy_input_single = tf.ones((1, *input_shape), dtype=tf.float32)
    
    # 1A. SYNFLOW (Conservación del Flujo)
    with tf.GradientTape() as tape_syn:
        tape_syn.watch(dummy_input_single)
        output_syn = model(dummy_input_single, training=False)
        loss_syn = tf.reduce_sum(output_syn) 
        
    grads_syn = tape_syn.gradient(loss_syn, model.trainable_weights)
    synflow_score = sum(
        float(tf.reduce_sum(tf.abs(g * w)).numpy()) 
        for g, w in zip(grads_syn, model.trainable_weights) if g is not None
    )

    # 1B. SNIP (Sensibilidad Estructural / Entrenabilidad)
    dummy_noise_target = tf.random.uniform(output_syn.shape, minval=0, maxval=1)
    with tf.GradientTape() as tape_snip:
        output_snip = model(dummy_input_single, training=False)
        loss_snip = tf.reduce_mean(tf.square(output_snip - dummy_noise_target))
        
    grads_snip = tape_snip.gradient(loss_snip, model.trainable_weights)
    snip_score = sum(
        float(tf.reduce_sum(tf.abs(g * w)).numpy()) 
        for g, w in zip(grads_snip, model.trainable_weights) if g is not None
    )

    # =====================================================================
    # FASE 2: PROXY DE COVARIANZA JACOBIANA (Expresividad Dimensional)
    # =====================================================================
    # Se requiere un lote > 1 para calcular la correlación cruzada
    batch_size = 16 
    dummy_batch = tf.random.normal((batch_size, *input_shape))
    
    # Inferencia puramente funcional (sin GradientTape para ahorrar VRAM)
    outputs_batch = model(dummy_batch, training=False)
    
    # Aplanamos los mapas de características espaciales de la U-Net a vectores 1D
    # [batch_size, H, W, C] -> [batch_size, H*W*C]
    outputs_flat = tf.reshape(outputs_batch, (batch_size, -1))
    
    # Centrado en la media (Zero-mean)
    outputs_flat = outputs_flat - tf.reduce_mean(outputs_flat, axis=0, keepdims=True)
    
    # Cálculo algebraico de la matriz de Covarianza (Tamaño: 16x16)
    num_features = tf.cast(tf.shape(outputs_flat)[1], tf.float32)
    cov_matrix = tf.matmul(outputs_flat, outputs_flat, transpose_b=True) / num_features
    
    # Estabilización numérica en la diagonal
    cov_matrix = cov_matrix + tf.eye(batch_size) * 1e-5
    
    # Extracción de valores singulares (SVD) para evaluar el volumen del espacio de características
    s, _, _ = tf.linalg.svd(cov_matrix)
    
    # El puntaje es la suma del logaritmo de los valores singulares (aproximación al log-determinante)
    jacobian_score = float(tf.reduce_sum(tf.math.log(s + 1e-5)).numpy())

    return synflow_score, snip_score, jacobian_score


def run_backfill(original_cache_path: str, enriched_cache_path: str):
    """
    Motor de ingesta y enriquecimiento retrospectivo.
    """
    print(f"[*] Iniciando Ingesta Retrospectiva Triple-ZCP...")
    print(f"[*] Base de datos fuente: {original_cache_path}")
    
    try:
        with open(original_cache_path, 'r', encoding='utf-8') as f:
            historical_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Archivo fuente no encontrado.")
        return

    enriched_data = {}
    total_archs = len(historical_data)
    start_time = time.time()

    for idx, (config_str, metrics) in enumerate(historical_data.items()):
        try:
            config = json.loads(config_str)
            
            # --- Instanciación Keras (Reemplazar con tu función importada) ---
            model = build_unet((256, 256, 1), **config)
            
            # Cálculo del ensamble
            synflow, snip, jacobian = compute_zero_cost_proxies(model)
            
            # Inyección de metadatos analíticos
            metrics['zcp_synflow'] = synflow
            metrics['zcp_snip'] = snip
            metrics['zcp_jacobian'] = jacobian
            
            enriched_data[config_str] = metrics
            
            if (idx + 1) % 10 == 0:
                print(f"    - Procesadas {idx + 1}/{total_archs} arquitecturas...")
                
        except Exception as e:
            print(f"    [WARN] Anomalía topológica en índice {idx}: {e}")
            
        finally:
            # Purga absoluta del grafo para blindar la VRAM
            if 'model' in locals(): del model
            tf.keras.backend.clear_session()
            gc.collect()

    with open(enriched_cache_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, sort_keys=True)
        
    elapsed = time.time() - start_time
    print(f"\n[*] Backfill completado en {elapsed:.2f} segundos.")
    print(f"[*] Base de datos maestra generada en: {enriched_cache_path}")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
    run_backfill("evaluation_cache.json", "zcp_evaluation_cache.json")
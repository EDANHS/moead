import gc
from pathlib import Path
import time
import traceback
import os
import numpy as np

from .data_manage import get_data_indices

def configure_device(use_gpu: bool):
    if use_gpu:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        try:
            import tensorflow as tf

            # PARCHE ESTRUCTURAL: DESACTIVAR LAYOUT OPTIMIZER
            tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    print(f"Advertencia al configurar VRAM: {e}")
                    
        except ImportError:
            pass
        print("--> GPU enabled (Layout Optimizer Desactivado para Estabilidad de Grafos Dinámicos)")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("--> GPU disabled, using CPU")

def evaluation_worker(queue, 
                      config, 
                      X_path, Y_path, 
                      input_shape, 
                      train_batch_size, 
                      val_batch_size, 
                      epochs, 
                      patience, 
                      max_trainable_params, 
                      verbose, 
                      use_gpu):
    """
    Proceso hijo aislado: Todo TensorFlow vive y muere aquí.
    """
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'    # Previene el secuestro absoluto de VRAM
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'        # Fuerza el crecimiento elástico
    os.environ['CUDA_CACHE_MAXSIZE'] = '4294967296'         # 4GB de caché para evitar recompilar PTX
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                # Silencia advertencias C++ no críticas
    raw_params = 0

    try:
        idx_train, idx_val, idx_test = get_data_indices(Path(X_path))
    except Exception as e:
        print(f"FATAL: {e}")
        return

    try:
        configure_device(use_gpu=use_gpu)

        # 1. Imports exclusivos del worker (Lazy Loading de TF)
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras import mixed_precision
        
        from moead.models import build_unet
        from moead.utils.tf_metrics import dice_coefficient, dice_loss

        # Configurar VRAM dinámicamente si es necesario (opcional pero recomendado)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                pass

        # Mixed Precision
        mixed_precision.set_global_policy("mixed_float16")

        # =================================================================
        # EL MOTOR ZERO-COPY (GENERADOR)
        # =================================================================
        # Abrimos un túnel directo de lectura al disco duro
        X_mmap = np.load(X_path, mmap_mode='r')
        Y_mmap = np.load(Y_path, mmap_mode='r')
        
        # Extraemos dinámicamente las formas exactas de una imagen para Keras
        shape_x = X_mmap.shape[1:] 
        shape_y = Y_mmap.shape[1:] 

        def build_tf_dataset(indices, batch_size, is_training):
            if len(indices) == 0:
                return None, 0
            
            # 1. Bucle infinito para que Keras nunca pida datos al vacío
            def data_generator():
                while True: 
                    idxs = np.copy(indices)
                    if is_training:
                        np.random.shuffle(idxs)
                    for i in idxs:
                        yield X_mmap[i], Y_mmap[i]

            ds = tf.data.Dataset.from_generator(
                data_generator,
                output_signature=(
                    tf.TensorSpec(shape=shape_x, dtype=tf.float32),
                    tf.TensorSpec(shape=shape_y, dtype=tf.float32)
                )
            )
            
            # 2. Calculamos los pasos matemáticos exactos (ceil para no perder el último lote)
            steps = max(1, int(np.ceil(len(indices) / batch_size)))
            
            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), steps

        # Construcción instantánea (ahora desempaquetamos los steps)
        train_ds, train_steps = build_tf_dataset(idx_train, train_batch_size, is_training=True)
        val_ds, val_steps = build_tf_dataset(idx_val, val_batch_size, is_training=False)
        
        test_ds = None
        test_steps = 0
        if len(idx_test) > 0:
            test_ds, test_steps = build_tf_dataset(idx_test, val_batch_size, is_training=False)

        # 3. Construcción del Modelo y Barrera Paramétrica
        model = build_unet(input_shape, **config)
        raw_params = model.count_params()

        if raw_params > max_trainable_params:
            queue.put({
                "success": False, "dice": 0.0, "params": raw_params,
                "elapsed": 0.0, "epochs": 0, "error": "OOM_PREVENTION"
            })
            return

        # 4. Compilación y Configuración
        optimizador_acumulativo = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            gradient_accumulation_steps=train_batch_size
        )

        model.compile(
            optimizer=optimizador_acumulativo,
            loss=dice_loss,
            metrics=[dice_coefficient],
            jit_compile=False
        )

        callbacks = []
        if patience > 0:
            callbacks.append(EarlyStopping(
                monitor='val_loss', patience=patience, 
                mode='min', restore_best_weights=True
            ))

        keras_fit_verbose = 2 if verbose >= 2 else 0

        start_time = time.time()
        history = model.fit(
            train_ds, 
            validation_data=val_ds,
            epochs=epochs, 
            steps_per_epoch=train_steps,  # <--- CONTROL ESTRICTO
            validation_steps=val_steps,   # <--- CONTROL ESTRICTO
            callbacks=callbacks,
            verbose=keras_fit_verbose
        )
        elapsed_time = time.time() - start_time
        epochs_registradas = len(history.history.get('loss', []))

        # 6. Evaluación
        eval_ds = test_ds if test_ds is not None else val_ds
        eval_steps = test_steps if test_ds is not None else val_steps
        
        keras_eval_verbose = 1 if verbose >= 2 else 0
        eval_results = model.evaluate(eval_ds, steps=eval_steps, verbose=keras_eval_verbose)
        
        # Asumiendo que eval_results = [loss, dice_coefficient]
        final_val_dice = float(eval_results[1])

        # Devolver resultados
        queue.put({
            "success": True,
            "dice": final_val_dice,
            "params": raw_params,
            "elapsed": elapsed_time,
            "epochs": epochs_registradas,
            "error": None
        })

    except Exception as e:
        error_msg = str(e)
        if "ResourceExhaustedError" in error_msg or "InternalError" in error_msg:
            error_type = "OOM"
        else:
            error_type = traceback.format_exc()
            
        queue.put({
            "success": False, "dice": 0.0, "params": raw_params if 'raw_params' in locals() else 0,
            "elapsed": 0.0, "epochs": 0, "error": error_type
        })
        
    finally:
        # 7. Purga de memoria en el worker antes de morir
        local_vars = ['model', 'history', 'optimizador_acumulativo', 'train_ds', 'val_ds', 'test_ds']
        for var in local_vars:
            if var in locals():
                del locals()[var]

        # Auditoría de VRAM segura
        try:
            import tensorflow as tf
            mem = tf.config.experimental.get_memory_info('GPU:0')
            print(f"    [GPU] current={mem['current']/1024**3:.2f}GB | peak={mem['peak']/1024**3:.2f}GB")
        except Exception:
            print("    [GPU] Información de memoria no disponible")
            pass 
        
        gc.collect()


def bounds_worker(q, input_shape, min_config, max_config):
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'    # Previene el secuestro absoluto de VRAM
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'        # Fuerza el crecimiento elástico
    os.environ['CUDA_CACHE_MAXSIZE'] = '4294967296'         # 4GB de caché para evitar recompilar PTX
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                # Silencia advertencias C++ no críticas
    
    try:
        from moead.models import build_unet
        import tensorflow as tf
        m_min = build_unet(input_shape, **min_config)
        p_min = m_min.count_params()
        del m_min
        
        m_max = build_unet(input_shape, **max_config)
        p_max = m_max.count_params()
        del m_max
        
        q.put((float(p_min), float(p_max)))
    except Exception:
        q.put((53.0, 350000000.0)) # Fallback genérico
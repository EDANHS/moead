"""Estimate GPU memory usage for U-Net model configurations and batch sizes.

This script builds the U-Net with the provided architecture parameters,
calculates parameter count and approximate activation memory, and optionally
measures real GPU memory usage with a warm-up run.

Usage example:
    python scripts/estimate_model_memory.py --depth 4 --initial_filters 32 --batch_size 8
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from moead.models import build_unet


def configure_tf(use_gpu: bool = True) -> None:
    if use_gpu:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        try:
            tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        except Exception:
            pass
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


def format_bytes(value: float) -> str:
    if value is None:
        return 'n/a'
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def get_gpu_memory_stats() -> dict[str, float] | None:
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return None

        gpu = gpus[0]
        info = tf.config.experimental.get_memory_info(gpu.name)

        current = float(info.get('current', 0))
        peak = float(info.get('peak', 0))
        limit = float(info.get('limit', 0)) if 'limit' in info else 0.0

        if current == 0 and peak == 0:
            fallback = get_gpu_memory_stats_nvidia_smi()
            if fallback is not None:
                return fallback

        result = {
            'current': current,
            'peak': peak,
        }

        if limit > 0:
            result['limit'] = limit

        return result

    except Exception:
        return get_gpu_memory_stats_nvidia_smi()


def get_gpu_memory_stats_nvidia_smi() -> dict[str, float] | None:
    import subprocess

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        line = result.stdout.strip().splitlines()[0]
        used, total = [int(x.strip()) for x in line.split(',')]
        return {
            'current': float(used) * 1024.0 ** 2,
            'limit': float(total) * 1024.0 ** 2,
            'peak': float(used) * 1024.0 ** 2,
        }
    except Exception:
        return None


def clear_tf_memory() -> None:
    tf.keras.backend.clear_session()
    gc.collect()
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.reset_memory_stats(gpu.name)
    except Exception:
        pass


def estimate_activation_memory(model: tf.keras.Model, batch_size: int) -> float:

    def shape_to_tuple(shape):
        if shape is None:
            return None

        if hasattr(shape, "as_list"):
            try:
                return tuple(shape.as_list())
            except Exception:
                pass

        try:
            return tuple(shape)
        except Exception:
            return None

    total_elements = 0
    seen_shapes = set()

    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        out = getattr(layer, "output", None)

        if out is None:
            continue

        shapes = []

        if isinstance(out, (list, tuple)):
            for item in out:
                if item is None:
                    continue

                if hasattr(item, "shape"):
                    shape = shape_to_tuple(item.shape)
                else:
                    shape = shape_to_tuple(item)

                if shape is not None:
                    shapes.append(shape)

        else:
            if hasattr(out, "shape"):
                shape = shape_to_tuple(out.shape)
            else:
                shape = shape_to_tuple(out)

            if shape is not None:
                shapes.append(shape)

        for shape in shapes:

            if shape is None:
                continue

            if len(shape) < 2:
                continue

            if any(dim is None for dim in shape[1:]):
                continue

            try:
                shape = tuple(int(dim) if dim is not None else dim for dim in shape)
            except Exception:
                continue

            if shape in seen_shapes:
                continue

            seen_shapes.add(shape)
            total_elements += int(np.prod(shape[1:]))

    return float(total_elements) * 4.0 * batch_size


def estimate_training_memory(weights_bytes: float, activation_bytes: float, overhead_ratio: float = 2.2) -> float:
    return weights_bytes + activation_bytes * overhead_ratio


def get_maximum_search_config() -> dict[str, Any]:
    return {
        'depth': 5,
        'initial_filters': 64,
        'kernel_size': (5, 5),
        'activation_name': 'Swish',
        'norm_type': 'Batch',
        'dropout_rate': 0.7,
        'use_bias': True,
        'pooling_type': 'Max',
        'upsample_type': 'TransposeConv',
    }


def get_gpu_total_bytes(default_gb: float = 12.0) -> float:
    stats = get_gpu_memory_stats()
    if stats is not None:
        return float(stats['limit'])
    return default_gb * 1024.0 ** 3


def perform_single_train_step(model: tf.keras.Model, batch: np.ndarray, labels: np.ndarray, optimizer=None):
    """Run one forward+backward step to measure GPU memory delta.

    Returns a tuple (success, before_bytes, after_bytes, error)
    """
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    try:
        before = get_gpu_memory_stats()
        if before is not None:
            before_bytes = float(before.get('current', 0))
        else:
            before_bytes = 0.0

        x = tf.convert_to_tensor(batch)
        y = tf.convert_to_tensor(labels)

        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_fn(y, preds)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        after = get_gpu_memory_stats()
        if after is not None:
            after_bytes = float(after.get('current', 0))
        else:
            after_bytes = 0.0

        return True, before_bytes, after_bytes, None

    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
        after = get_gpu_memory_stats()
        after_bytes = float(after.get('current', 0)) if after is not None else 0.0
        return False, before_bytes, after_bytes, e
    except Exception as e:
        after = get_gpu_memory_stats()
        after_bytes = float(after.get('current', 0)) if after is not None else 0.0
        return False, before_bytes, after_bytes, e


def load_real_batch(x_path: str | Path, y_path: str | Path, batch_size: int):
    x_path = Path(x_path)
    y_path = Path(y_path)
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"No se encontraron archivos: {x_path} or {y_path}")

    X = np.load(x_path)
    Y = np.load(y_path)

    if X.ndim == 3:
        X = X[..., None]
    if Y.ndim == 3:
        Y = Y[..., None]

    assert X.shape[0] == Y.shape[0], 'X and Y must have same number of samples'
    n = X.shape[0]
    if n < batch_size:
        raise ValueError(f'Not enough samples ({n}) for batch_size {batch_size}')

    idx = np.random.choice(n, batch_size, replace=False)
    batch = X[idx].astype(np.float32)
    labels = Y[idx].astype(np.float32)
    return batch, labels


def load_real_dataset(x_path: str | Path, y_path: str | Path, max_samples: int):
    x_path = Path(x_path)
    y_path = Path(y_path)
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"No se encontraron archivos: {x_path} or {y_path}")

    X = np.load(x_path)
    Y = np.load(y_path)

    if X.ndim == 3:
        X = X[..., None]
    if Y.ndim == 3:
        Y = Y[..., None]

    assert X.shape[0] == Y.shape[0], 'X and Y must have same number of samples'
    n = X.shape[0]
    if n == 0:
        raise ValueError('No hay muestras en el dataset')

    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        X = X[idx]
        Y = Y[idx]

    return X.astype(np.float32), Y.astype(np.float32)


def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, is_training: bool = True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if is_training:
        ds = ds.shuffle(buffer_size=min(128, len(x)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds


def train_one_epoch(model: tf.keras.Model, x: np.ndarray, y: np.ndarray, batch_size: int):
    ds = make_tf_dataset(x, y, batch_size, is_training=True)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    try:
        before = get_gpu_memory_stats()
        before_bytes = float(before.get('current', 0)) if before is not None else 0.0
        history = model.fit(ds, epochs=1, verbose=1)
        after = get_gpu_memory_stats()
        after_bytes = float(after.get('current', 0)) if after is not None else 0.0
        return True, before_bytes, after_bytes, None, history
    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
        after = get_gpu_memory_stats()
        after_bytes = float(after.get('current', 0)) if after is not None else 0.0
        return False, before_bytes, after_bytes, e, None
    except Exception as e:
        after = get_gpu_memory_stats()
        after_bytes = float(after.get('current', 0)) if after is not None else 0.0
        return False, before_bytes, after_bytes, e, None


def find_dataset_files(key: str) -> tuple[Path, Path] | None:
    root = Path(PROJECT_ROOT)
    candidates = [
        (root / 'dataset_optimizado_256' / f'X_train_{key}.npy', root / 'dataset_optimizado_256' / f'Y_mask_{key}.npy'),
        (root / 'data' / f'X_train_{key}_5k.npy', root / 'data' / f'Y_train_{key}_5k.npy'),
    ]
    for x_path, y_path in candidates:
        if x_path.exists() and y_path.exists():
            return x_path, y_path
    return None


def auto_reduce_from_max(
    input_shape: tuple[int, int, int],
    batch_size: int,
    reserve_ratio: float,
    measure_gpu: bool,
) -> None:
    config = get_maximum_search_config()
    current_batch = batch_size
    current_filters = config['initial_filters']
    current_depth = config['depth']
    target_gpu_bytes = get_gpu_total_bytes()
    reserved = target_gpu_bytes * reserve_ratio
    target_budget = target_gpu_bytes - reserved

    print('\n--- Ajuste automático desde configuración máxima ---')
    print(f"GPU objetivo estimada: {format_bytes(target_budget)} (reserva {reserve_ratio*100:.0f}% = {format_bytes(reserved)})")
    print(f"Configuración inicial máxima: depth={current_depth}, filters={current_filters}, kernel={config['kernel_size']}, dropout={config['dropout_rate']}, batch_size={current_batch}\n")

    while True:
        clear_tf_memory()
        model = build_unet(
            input_shape=input_shape,
            depth=current_depth,
            initial_filters=current_filters,
            kernel_size=config['kernel_size'],
            activation_name=config['activation_name'],
            norm_type=config['norm_type'],
            dropout_rate=config['dropout_rate'],
            use_bias=config['use_bias'],
            pooling_type=config['pooling_type'],
            upsample_type=config['upsample_type'],
        )

        params = model.count_params()
        weights_memory = float(params) * 4.0
        activations_memory = estimate_activation_memory(model, current_batch)
        training_memory = estimate_training_memory(weights_memory, activations_memory)
        print(f"Evaluando: depth={current_depth}, filters={current_filters}, batch_size={current_batch}, params={params:,}")
        print(f"  Memoria estimada: {format_bytes(training_memory)}")

        if training_memory <= target_budget:
            print('  -> Configuración válida encontrada')
            build_and_measure(
                input_shape=input_shape,
                depth=current_depth,
                initial_filters=current_filters,
                kernel_size=config['kernel_size'],
                activation_name=config['activation_name'],
                norm_type=config['norm_type'],
                dropout_rate=config['dropout_rate'],
                use_bias=config['use_bias'],
                pooling_type=config['pooling_type'],
                upsample_type=config['upsample_type'],
                batch_size=current_batch,
                reserve_ratio=reserve_ratio,
                measure_gpu=measure_gpu,
            )
            return

        if current_batch > 1:
            current_batch = max(1, current_batch // 2)
            print('  Reduciendo batch_size para bajar memoria...')
        elif current_filters > 2:
            current_filters = max(2, current_filters - 2)
            print('  Batch mínimo alcanzado; reduciendo initial_filters...')
        elif current_depth > 1:
            current_depth -= 1
            print('  Filtros mínimos alcanzados; reduciendo depth...')
        else:
            print('  No fue posible encontrar una configuración ajustada dentro del presupuesto estimado.')
            print('  Última configuración probada:')
            print(f"    depth={current_depth}, filters={current_filters}, batch_size={current_batch}, params={params:,}")
            print(f"    Memoria estimada: {format_bytes(training_memory)}")
            return

        print('  Reintentando con valores menores...\n')
        del model


def build_and_measure(
    input_shape: tuple[int, int, int],
    depth: int,
    initial_filters: int,
    kernel_size: tuple[int, int],
    activation_name: str,
    norm_type: str,
    dropout_rate: float,
    use_bias: bool,
    pooling_type: str,
    upsample_type: str,
    batch_size: int,
    reserve_ratio: float,
    measure_gpu: bool,
) -> None:
    print("\n--- Estimación de memoria del modelo ---")
    print(f"input_shape: {input_shape}")
    print(f"depth: {depth}, initial_filters: {initial_filters}, kernel_size: {kernel_size}")
    print(f"activation: {activation_name}, norm: {norm_type}, upsample: {upsample_type}")
    print(f"dropout: {dropout_rate:.4f}, use_bias: {use_bias}, batch_size: {batch_size}")

    clear_tf_memory()
    model = build_unet(
        input_shape=input_shape,
        depth=depth,
        initial_filters=initial_filters,
        kernel_size=kernel_size,
        activation_name=activation_name,
        norm_type=norm_type,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        pooling_type=pooling_type,
        upsample_type=upsample_type,
    )

    params = model.count_params()
    weights_memory = float(params) * 4.0
    activations_memory = estimate_activation_memory(model, batch_size)
    training_memory = estimate_training_memory(weights_memory, activations_memory)
    target_gpu_mem = None
    reserved_mem = None
    available_mem = None

    print(f"\nParámetros totales: {params:,}")
    print(f"Peso aproximado de pesos: {format_bytes(weights_memory)}")
    print(f"Memoria de activaciones (aprox. forward por batch): {format_bytes(activations_memory)}")
    print(f"Estimación de memoria total para entrenamiento: {format_bytes(training_memory)}")

    if measure_gpu:
        print("\nMidiendo memoria GPU real (si está disponible)...")
        gpu_stats = get_gpu_memory_stats()
        if gpu_stats is not None:
            available_mem = float(gpu_stats['limit'])
            reserved_mem = available_mem * reserve_ratio
            target_gpu_mem = available_mem - reserved_mem
            print(f"GPU total: {format_bytes(available_mem)}")
            print(f"Objetivo conservador (reserve {reserve_ratio*100:.0f}%): {format_bytes(target_gpu_mem)}")

            batch = np.zeros((batch_size, *input_shape), dtype=np.float32)
            clear_tf_memory()
            tf_model = build_unet(
                input_shape=input_shape,
                depth=depth,
                initial_filters=initial_filters,
                kernel_size=kernel_size,
                activation_name=activation_name,
                norm_type=norm_type,
                dropout_rate=dropout_rate,
                use_bias=use_bias,
                pooling_type=pooling_type,
                upsample_type=upsample_type,
            )
            tf_model.compile(optimizer='adam', loss='binary_crossentropy')
            clear_tf_memory()
            tf.keras.backend.clear_session()
            gc.collect()
            try:
                tf.config.experimental.reset_memory_stats(tf.config.list_physical_devices('GPU')[0].name)
            except Exception:
                pass

            before = get_gpu_memory_stats()
            if before is not None:
                print(f"Memoria antes del warm-up: {format_bytes(before['current'])}")
            try:
                _ = tf_model.predict(batch, batch_size=batch_size, verbose=0)
            except Exception as e:
                print(f"  Error en inferencia medida: {e}")
                before = get_gpu_memory_stats()
                if before is not None:
                    print(f"Memoria tras error: {format_bytes(before['current'])}")
            after = get_gpu_memory_stats()
            if after is not None and before is not None:
                delta = after['current'] - before['current']
                print(f"Consumo GPU estimado tras predict: {format_bytes(delta)}")

            try:
                tf.config.experimental.reset_memory_stats(tf.config.list_physical_devices('GPU')[0].name)
            except Exception:
                pass
            clear_tf_memory()
        else:
            print("No hay GPU disponible o no se pudo leer la memoria GPU.")

    free_margin = None
    if available_mem is not None:
        free_margin = available_mem - training_memory
        print(f"\nMargen libre estimado (GPU total menos memoria estimada de entrenamiento): {format_bytes(free_margin)}")
        if free_margin <= 0:
            print("ADVERTENCIA: la estimación indica que el modelo puede superar la memoria disponible de GPU.")

    print("\n--- Recomendaciones ---")
    if free_margin is not None and free_margin < 0:
        print("  - Reduce batch_size o initial_filters para bajar el uso de memoria.")
        print("  - Considera reserve_ratio mayor si trabajas con GPU cercana al límite.")
    else:
        print("  - El modelo parece estar dentro de un rango estimado, pero verifica con un run real.")
    print("  - Usa `TF_GPU_ALLOCATOR=cuda_malloc_async` y `tf.config.experimental.set_memory_growth` para mejorar la estabilidad.")


def parse_kernel(kernel_str: str) -> tuple[int, int]:
    if ',' in kernel_str:
        parts = [int(x.strip()) for x in kernel_str.split(',') if x.strip()]
        if len(parts) == 2:
            return (parts[0], parts[1])
    return (int(kernel_str), int(kernel_str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate U-Net GPU memory usage for model configuration.")
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--initial_filters', type=int, default=32)
    parser.add_argument('--kernel_size', type=str, default='3')
    parser.add_argument('--activation_name', type=str, default='ReLU')
    parser.add_argument('--norm_type', type=str, default='Batch')
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--use_bias', action='store_true')
    parser.add_argument('--pooling_type', type=str, default='Max')
    parser.add_argument('--upsample_type', type=str, default='TransposeConv')
    parser.add_argument('--input_height', type=int, default=256)
    parser.add_argument('--input_width', type=int, default=256)
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--reserve_ratio', type=float, default=0.15, help='Fraction of GPU memory to reserve for safety')
    parser.add_argument('--no_gpu_measure', action='store_true', help='Skip actual GPU memory measurement')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Enable GPU for the measurement')
    parser.add_argument('--use_max_config', action='store_true', help='Start from the maximum memory configuration')
    parser.add_argument('--auto_reduce', action='store_true', help='Reduce batch/filters/depth from max config until memory fits')
    parser.add_argument('--measure_train_step', action='store_true', help='Run one forward+backward step to measure peak GPU memory')
    parser.add_argument('--train_one_epoch', action='store_true', help='Run a one-epoch training test on a small dataset')
    parser.add_argument('--train_samples', type=int, default=16, help='Max number of samples to use for one epoch training test')
    parser.add_argument('--use_real_data', action='store_true', help='Load a real batch from dataset files instead of zeros')
    parser.add_argument('--x_np', type=str, default=None, help='Path to X numpy file to use for real batch')
    parser.add_argument('--y_np', type=str, default=None, help='Path to Y numpy file to use for real batch')
    parser.add_argument('--dataset_key', type=str, default='ctv', help='Short name for dataset files under dataset_optimizado_256 (e.g. ctv, vejiga)')
    args = parser.parse_args()

    configure_tf(args.use_gpu)
    kernel_size = parse_kernel(args.kernel_size)
    if args.measure_train_step or args.train_one_epoch:
        cfg = get_maximum_search_config() if args.use_max_config else {
            'depth': args.depth,
            'initial_filters': args.initial_filters,
            'kernel_size': kernel_size,
            'activation_name': args.activation_name,
            'norm_type': args.norm_type,
            'dropout_rate': args.dropout_rate,
            'use_bias': args.use_bias,
            'pooling_type': args.pooling_type,
            'upsample_type': args.upsample_type,
        }

        clear_tf_memory()
        model = build_unet(
            input_shape=(args.input_height, args.input_width, args.input_channels),
            depth=cfg['depth'],
            initial_filters=cfg['initial_filters'],
            kernel_size=cfg['kernel_size'],
            activation_name=cfg['activation_name'],
            norm_type=cfg['norm_type'],
            dropout_rate=cfg['dropout_rate'],
            use_bias=cfg['use_bias'],
            pooling_type=cfg['pooling_type'],
            upsample_type=cfg['upsample_type'],
        )

        if args.use_real_data:
            if args.x_np and args.y_np:
                try:
                    batch, labels = load_real_batch(args.x_np, args.y_np, args.batch_size)
                except Exception as e:
                    print(f"Error cargando datos desde rutas proporcionadas: {e}")
                    sys.exit(1)
            else:
                found = find_dataset_files(args.dataset_key)
                if found is None:
                    print('No se encontraron archivos de dataset automáticos. Usa --x_np y --y_np para rutas explícitas.')
                    sys.exit(1)

                try:
                    if args.train_one_epoch:
                        X, Y = load_real_dataset(found[0], found[1], args.train_samples)
                    else:
                        batch, labels = load_real_batch(found[0], found[1], args.batch_size)
                except Exception as e:
                    print(f'Error cargando datos reales: {e}')
                    sys.exit(1)
        else:
            batch = np.zeros((args.batch_size, args.input_height, args.input_width, args.input_channels), dtype=np.float32)
            labels = np.zeros_like(batch)
            if args.train_one_epoch:
                X = np.tile(batch, (max(1, args.train_samples // args.batch_size), 1, 1, 1))
                Y = np.tile(labels, (max(1, args.train_samples // args.batch_size), 1, 1, 1))

        if args.measure_train_step:
            print('\nEjecutando un único paso de entrenamiento (forward+backward) para medir memoria...')
            success, before_b, after_b, err = perform_single_train_step(model, batch, labels)
            print(f"Memoria antes: {format_bytes(before_b)} | Memoria después: {format_bytes(after_b)} | Delta: {format_bytes(max(0, after_b-before_b))}")
            if not success:
                print(f"Error durante el paso de entrenamiento: {err}")
            sys.exit(0)

        if args.train_one_epoch:
            print('\nEjecutando un epoch de entrenamiento para testear caída...')
            success, before_b, after_b, err, history = train_one_epoch(model, X, Y, args.batch_size)
            print(f"Memoria antes: {format_bytes(before_b)} | Memoria después: {format_bytes(after_b)} | Delta: {format_bytes(max(0, after_b-before_b))}")
            if success:
                print('Entrenamiento completado sin excepción durante el primer epoch.')
                print(f'Historia: {history.history if history is not None else {{}}}')
            else:
                print(f"Error durante el entrenamiento de un epoch: {err}")
            sys.exit(0)
    if args.use_max_config:
        if args.auto_reduce:
            auto_reduce_from_max(
                input_shape=(args.input_height, args.input_width, args.input_channels),
                batch_size=args.batch_size,
                reserve_ratio=args.reserve_ratio,
                measure_gpu=not args.no_gpu_measure,
            )
        else:
            max_config = get_maximum_search_config()
            build_and_measure(
                input_shape=(args.input_height, args.input_width, args.input_channels),
                depth=max_config['depth'],
                initial_filters=max_config['initial_filters'],
                kernel_size=max_config['kernel_size'],
                activation_name=max_config['activation_name'],
                norm_type=max_config['norm_type'],
                dropout_rate=max_config['dropout_rate'],
                use_bias=max_config['use_bias'],
                pooling_type=max_config['pooling_type'],
                upsample_type=max_config['upsample_type'],
                batch_size=args.batch_size,
                reserve_ratio=args.reserve_ratio,
                measure_gpu=not args.no_gpu_measure,
            )
    else:
        build_and_measure(
            input_shape=(args.input_height, args.input_width, args.input_channels),
            depth=args.depth,
            initial_filters=args.initial_filters,
            kernel_size=kernel_size,
            activation_name=args.activation_name,
            norm_type=args.norm_type,
            dropout_rate=args.dropout_rate,
            use_bias=args.use_bias,
            pooling_type=args.pooling_type,
            upsample_type=args.upsample_type,
            batch_size=args.batch_size,
            reserve_ratio=args.reserve_ratio,
            measure_gpu=not args.no_gpu_measure,
        )


if __name__ == '__main__':
    main()

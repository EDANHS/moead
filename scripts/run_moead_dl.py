"""
Small runner script to execute MOEAD_DL for DL hyperparameter optimization.
OPTIMIZED RUNNER: Cleans up imports and relies on installed package.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# ==============================================================================
# 🛑 PARCHE NUCLEAR V2 PARA RTX 5080 (BLACKWELL)
# ==============================================================================
import os

# 1. Desactivar XLA/JIT (Causa principal de INVALID_PTX)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0"

# 2. Desactivar TensorFloat-32 (TF32)
# Las tarjetas nuevas usan esto por defecto para acelerar, pero en TF viejo falla.
os.environ["NVIDIA_TF32_OVERRIDE"] = "0" 

# 3. Logs limpios
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
# ==============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from moead.algorithms import MOEAD_DL
from moead.evolutionary_operator import DifferentialEvolution
from moead.scalarizations import PBI
from moead.problems import DLProblem

CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent 

def parse_args():
    p = argparse.ArgumentParser(description="Run MOEAD_DL optimization")
    p.add_argument('--use-gpu', action='store_true', default=True, help='Enable GPU')
    p.add_argument('--n_generations', type=int, default=200, help='Generations')
    p.add_argument('--h_divisions', type=int, default=149, help='H divisions')
    p.add_argument('--n_neighbors', type=int, default=30, help='Neighbors')
    p.add_argument('--n_r', type=int, default=2, help='Max replacements')
    p.add_argument('--log', type=str, default='moead_dl_log.json', help='Log file')
    p.add_argument('--checkpoint', type=str, default='moead_dl_checkpoint.pkl', help='Checkpoint')
    return p.parse_args()

def configure_device(use_gpu: bool):
    """Configura la GPU con ESTABILIDAD MÁXIMA para RTX 5080."""
    if use_gpu:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        print('--> GPU mode requested.')
        try:
            import tensorflow as tf
            
            # 1. Configurar Memory Growth (Vital para evitar INVALID_HANDLE)
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"--> Memory Growth activado para: {gpus}")
                except RuntimeError as e:
                    print(f"--> Error configurando Memory Growth: {e}")

            # 2. Desactivamos Mixed Precision por ahora para estabilidad
            # from tensorflow.keras import mixed_precision
            # policy = mixed_precision.Policy('mixed_float16')
            # mixed_precision.set_global_policy(policy)
            
            print("--> Modo: Float32 Standard (Estabilidad activada)")

        except ImportError:
            print("--> Error importando TensorFlow para configuración de GPU.")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print('--> CPU mode: disabling CUDA_VISIBLE_DEVICES')

def load_data(root_path: Path) -> tuple:
    """Carga datos directamente desde la estructura del proyecto."""
    data_dir = root_path / 'data' 
    path_x = data_dir / "images_train.npy"
    path_y = data_dir / "masks_train.npy"
    
    print(f"--> Cargando datos desde: {data_dir}")
    
    if not path_x.exists() or not path_y.exists():
        raise FileNotFoundError(f"No se encontraron los archivos .npy en {data_dir}")

    try:
        # --- CAMBIO CRÍTICO AQUÍ ---
        # Convertimos a float32 INMEDIATAMENTE para evitar Casts en la GPU
        X = np.load(path_x).astype(np.float32)
        Y = np.load(path_y).astype(np.float32)
        print(f"--> Datos cargados y convertidos a float32. Shape X: {X.shape}")
    except Exception as e:
        raise RuntimeError(f"Error cargando .npy: {e}")
    
    X_t, X_v, Y_t, Y_v = train_test_split(X, Y, test_size=0.2, random_state=42)
    del X, Y
    return X_t, Y_t, X_v, Y_v

def plot_front(archive, out_path: Path):
    if not archive: return
    f1 = [s.objectives[0] for s in archive]
    f2 = [s.objectives[1] for s in archive]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(f1, f2, s=20, c='blue', alpha=0.6)
    plt.xlabel('Dice Loss (Minimizar)')
    plt.ylabel('Norm Params (Minimizar)')
    plt.title(f'MOEAD-DL Pareto Front ({len(archive)} soluciones)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_history(history: dict, out_path: Path):
    try:
        if hasattr(history, 'get_history'):
            h = history.get_history()
        else:
            h = history
            
        z_star = np.array(h.get('z_star_per_gen', []))
        archive_sizes = h.get('archive_size_per_gen', [])
        
        if len(archive_sizes) == 0: return

        gens = np.arange(len(archive_sizes))
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        if z_star.shape[0] > 0 and z_star.shape[1] >= 2:
            plt.plot(gens, z_star[:, 0], label='Best Dice Loss found')
            plt.plot(gens, z_star[:, 1], label='Best Size found')
        plt.title("Evolución del Punto Ideal (Z*)")
        plt.legend(); plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(gens, archive_sizes, 'r-o', label='Archive Size')
        plt.xlabel("Generación")
        plt.title("Tamaño del Archivo Externo")
        plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Error ploteando historial: {e}")

def main():
    args = parse_args()
    configure_device(args.use_gpu)
    
    try:
        X_train, Y_train, X_val, Y_val = load_data(PROJECT_ROOT)
    except Exception as e:
        print(f"FATAL: {e}")
        return

    problem = DLProblem(X_train, Y_train, X_val, Y_val, batch_size=32)
    
    scalarization = PBI()
    evo_op = DifferentialEvolution(F=0.5, CR=0.6, selection_prob=0.9)

    img_dir = PROJECT_ROOT / 'images'
    img_dir.mkdir(exist_ok=True, parents=True)
    
    log_path = PROJECT_ROOT / args.log
    chk_path = PROJECT_ROOT / args.checkpoint

    print(f"--> Iniciando MOEAD_DL: Gens={args.n_generations}, Vecinos={args.n_neighbors}")
    
    moead = MOEAD_DL(
        problem=problem,
        scalarization=scalarization,
        evolutionary_op=evo_op,
        h_divisions=args.h_divisions,
        n_neighbors=args.n_neighbors,
        n_generations=args.n_generations,
        n_r=args.n_r,
        log_filename=str(log_path),
        checkpoint_file=str(chk_path),
    )

    archive, history = moead.run()

    plot_front(archive, img_dir / 'moead_dl_front.png')
    plot_history(history, img_dir / 'moead_dl_history.png')
    
    print("--> Ejecución finalizada con éxito.")

if __name__ == '__main__':
    main()
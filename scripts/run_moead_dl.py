"""
Small runner script to execute MOEAD_DL for DL hyperparameter optimization.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS (PARCHE NUCLEAR)
# ==============================================================================

# Definimos las rutas clave como CONSTANTES GLOBALES usando pathlib (más robusto)
CURRENT_SCRIPT = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_SCRIPT.parent
PROJECT_ROOT = SCRIPTS_DIR.parent          # C:\...\MOEAD
INNER_LIB = PROJECT_ROOT / 'moead'         # C:\...\MOEAD\moead

print(f"--> Configurando entorno...")
print(f"    Raíz del proyecto: {PROJECT_ROOT}")

# A. Añadir la raíz (para: from moead.algorithms import ...)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# B. Añadir la librería interna (para: import algorithms)
if str(INNER_LIB) not in sys.path:
    sys.path.insert(0, str(INNER_LIB))

# C. PARCHE NUCLEAR: Añadir TODAS las subcarpetas de 'moead' al path.
# Esto soluciona que 'algorithms.py' no encuentre 'solutions.py' si están en carpetas separadas.
if INNER_LIB.exists():
    for item in INNER_LIB.iterdir():
        if item.is_dir():
            if str(item) not in sys.path:
                sys.path.insert(0, str(item))
                # print(f"    [Parche] Subcarpeta añadida: {item.name}")

# ==============================================================================

PROJECT_ROOT = "C:/Users/thoma/Documents/Proyectos/MOEAD/moead/colabs/dataset_npy_vejiga/"

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def parse_args():
    p = argparse.ArgumentParser(description="Run MOEAD_DL optimization")
    p.add_argument('--use-gpu', action='store_true', help='Enable GPU')
    p.add_argument('--n_generations', type=int, default=5, help='Generations')
    p.add_argument('--h_divisions', type=int, default=3, help='H divisions')
    p.add_argument('--n_neighbors', type=int, default=2, help='Neighbors')
    p.add_argument('--n_r', type=int, default=1, help='Max replacements')
    p.add_argument('--log', type=str, default='moead_dl_log.json', help='Log file')
    p.add_argument('--checkpoint', type=str, default='moead_dl_checkpoint.pkl', help='Checkpoint')
    return p.parse_args()

def configure_device(use_gpu: bool):
    if use_gpu:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        print('GPU mode requested: permitting CUDA-visible devices')
        try:
            import tensorflow as tf
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print('--> Mixed Precision (float16) activado.')
        except ImportError:
            pass
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print('CPU mode: disabling CUDA_VISIBLE_DEVICES')

def load_data_try_npy() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga datos usando la constante global PROJECT_ROOT."""
    path_x = f"{PROJECT_ROOT}/images_train.npy"
    path_y = f"{PROJECT_ROOT}/masks_train.npy"
    try:
        X = np.load(path_x)
        Y = np.load(path_y)
    except Exception as e:
        raise RuntimeError(f"No se pudieron cargar los datos .npy: {e}")
    
    X_t, X_v, Y_t, Y_v = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X_t, Y_t, X_v, Y_v

def plot_front(archive, out_path: Path):
    if not archive: return
    f1 = [s.objectives[0] for s in archive]
    f2 = [s.objectives[1] for s in archive]
    plt.figure(figsize=(8, 6))
    plt.scatter(f1, f2, s=20)
    plt.xlabel('Dice Loss (Minimizar)')
    plt.ylabel('Norm Params (Minimizar)')
    plt.title('MOEAD-DL Pareto Front')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_history(history: dict, out_path: Path):
    try:
        z_star = np.array(history['z_star_per_gen'])
        archive_sizes = history['archive_size_per_gen']
    except Exception:
        return

    gens = np.arange(len(archive_sizes))
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    if z_star.shape[1] >= 2:
        plt.plot(gens, z_star[:, 0], label='z*_1 (Dice)')
        plt.plot(gens, z_star[:, 1], label='z*_2 (Params)')
    plt.legend(); plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(gens, archive_sizes, 'r-o', label='Archive size')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    args = parse_args()
    configure_device(args.use_gpu)
    
    print("--> Importando librerías moead...")
    try:
        # Intentamos importar con el nombre completo
        from moead.algorithms import MOEAD_DL
        from moead.evolutionary_operator import DifferentialEvolution
        from moead.scalarizations import PBI
        from moead.problems import DLProblem
    except ImportError:
        # Fallback por si los parches hicieron que 'moead' sea redundante
        try:
            from algorithms import MOEAD_DL
            from evolutionary_operator import DifferentialEvolution
            from scalarizations import PBI
            from problems import DLProblem
            print("--> Imports resolved using direct module names.")
        except ImportError as e:
            print(f"\n[ERROR FATAL DE IMPORTACIÓN] {e}")
            print(f"PYTHONPATH: {sys.path}")
            sys.exit(1)

    # Cargar datos
    X_train, Y_train, X_val, Y_val = load_data_try_npy()
    
    # IMPORTANTE: Calcular input_shape
    if len(X_train.shape) >= 2:
        input_shape = X_train.shape[1:]
    else:
        input_shape = (32, 32, 1) # Fallback seguro
        
    print(f"--> Input Shape detectado: {input_shape}")

    # Configurar problema
    problem = DLProblem(X_train, Y_train, X_val, Y_val)
    
    scalarization = PBI()
    evo_op = DifferentialEvolution(F=0.5, CR=0.9, selection_prob=0.9)

    img_dir = Path('./images')
    img_dir.mkdir(exist_ok=True, parents=True)

    log_file = Path(args.log)
    checkpoint_file = Path(args.checkpoint)

    print(f"--> Iniciando MOEAD_DL con {args.n_generations} generaciones...")
    moead = MOEAD_DL(
        problem=problem,
        scalarization=scalarization,
        evolutionary_op=evo_op,
        h_divisions=args.h_divisions,
        n_neighbors=args.n_neighbors,
        n_generations=args.n_generations,
        n_r=args.n_r,
        log_filename=str(log_file),
        checkpoint_file=str(checkpoint_file),
    )

    archive, history = moead.run()

    front_path = img_dir / 'moead_dl_front.png'
    plot_front(archive, front_path)
    print(f'Front image saved to: {front_path}')

    history_path = img_dir / 'moead_dl_history.png'
    if isinstance(history, dict):
        plot_history(history, history_path)
    elif hasattr(history, 'get_history'):
        try:
            plot_history(history.get_history(), history_path)
        except: pass

    summary_json = f"{checkpoint_file}.json"
    print(f'Checkpoint: {checkpoint_file}')
    print(f'Log: {log_file}')

if __name__ == '__main__':
    main()
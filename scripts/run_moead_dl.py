"""
Small runner script to execute MOEAD_DL for DL hyperparameter optimization.
OPTIMIZED RUNNER: Cleans up imports and relies on installed package.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend sin interfaz gráfica (ideal para servidores/background)
import matplotlib.pyplot as plt

# --- IMPORTS DE TU LIBRERÍA (Ya no requieren parches sys.path) ---
from sklearn.model_selection import train_test_split
from moead.algorithms import MOEAD_DL
from moead.evolutionary_operator import DifferentialEvolution
from moead.scalarizations import PBI
from moead.problems import DLProblem

# Configuración de rutas relativas simples
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent # Asumiendo scripts/run.py -> root es ../

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
    """Configura la GPU y Mixed Precision para RTX 5080."""
    if use_gpu:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        print('--> GPU mode requested.')
        try:
            import tensorflow as tf
            # Verificar visibilidad física
            gpus = tf.config.list_physical_devices('GPU')
            print(f"--> TensorFlow ve {len(gpus)} GPUs: {gpus}")
            
            # Activar Mixed Precision (Vital para RTX 5080)
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print('--> Mixed Precision (float16) activado.')
        except ImportError:
            print("--> Error importando TensorFlow para configuración de GPU.")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print('--> CPU mode: disabling CUDA_VISIBLE_DEVICES')

def load_data(root_path: Path) -> tuple:
    """Carga datos directamente desde la estructura del proyecto."""
    # Ajusta esta ruta si tus datos están en otro lado (ej. 'data' o 'dataset')
    data_dir = root_path / 'data' 
    
    path_x = data_dir / "images_train.npy"
    path_y = data_dir / "masks_train.npy"
    
    print(f"--> Cargando datos desde: {data_dir}")
    
    if not path_x.exists() or not path_y.exists():
        raise FileNotFoundError(f"No se encontraron los archivos .npy en {data_dir}")

    try:
        X = np.load(path_x)
        Y = np.load(path_y)
        print(f"--> Datos cargados. Shape X: {X.shape}")
    except Exception as e:
        raise RuntimeError(f"Error cargando .npy: {e}")
    
    # Split
    X_t, X_v, Y_t, Y_v = train_test_split(X, Y, test_size=0.2, random_state=42)
    del X, Y # Liberar RAM
    return X_t, Y_t, X_v, Y_v

def plot_front(archive, out_path: Path):
    if not archive: return
    # Extraer objetivos. Ojo: asegúrate que tus objetivos sean 1-Dice y NormParams
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
    # Lógica robusta para graficar historial
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
        
        # Subplot 1: Punto Ideal (Z*)
        plt.subplot(2, 1, 1)
        if z_star.shape[0] > 0 and z_star.shape[1] >= 2:
            plt.plot(gens, z_star[:, 0], label='Best Dice Loss found')
            plt.plot(gens, z_star[:, 1], label='Best Size found')
        plt.title("Evolución del Punto Ideal (Z*)")
        plt.legend(); plt.grid(True)
        
        # Subplot 2: Tamaño del Archivo
        plt.subplot(2, 1, 2)
        plt.plot(gens, archive_sizes, 'r-o', label='Archive Size')
        plt.xlabel("Generación")
        plt.title("Tamaño del Archivo Externo (Soluciones No Dominadas)")
        plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Error ploteando historial: {e}")

def main():
    args = parse_args()
    configure_device(args.use_gpu)
    
    # 1. Cargar Datos
    try:
        X_train, Y_train, X_val, Y_val = load_data(PROJECT_ROOT)
    except Exception as e:
        print(f"FATAL: {e}")
        return

    # 2. Configurar Problema (Aquí ocurre la magia de tf.data)
    # Pasamos batch_size=32 o 64 para la RTX 5080
    problem = DLProblem(X_train, Y_train, X_val, Y_val, batch_size=32)
    
    # 3. Configurar Algoritmo
    scalarization = PBI()
    # CR (Crossover Rate) alto (0.9) suele ser mejor para DE en problemas continuos
    evo_op = DifferentialEvolution(F=0.5, CR=0.6, selection_prob=0.9)

    # Rutas de salida
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

    # 4. Ejecutar
    archive, history = moead.run()

    # 5. Guardar Resultados
    plot_front(archive, img_dir / 'moead_dl_front.png')
    plot_history(history, img_dir / 'moead_dl_history.png')
    
    print("--> Ejecución finalizada con éxito.")

if __name__ == '__main__':
    main()
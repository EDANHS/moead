"""
Runner clásico de MOEAD_DL sin DifferentialEvolution.

Este script usa la reproducción clásica de MOEA/D basada en SBX + mutación
polinómica (clásico) y guarda metadata estructurada sobre la ejecución.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Redirigir importaciones a la raíz del proyecto
from moead.algorithms import MOEAD_DL
from moead.crossovers import SBXCrossover
from moead.evolutionary_operator.CrossoverMutation import CrossoverMutation
from moead.mutations.PolynomialMutation import PolynomialMutation
from moead.scalarizations import PBI
from moead.problems import DLProblem


def parse_args():
    p = argparse.ArgumentParser(description="Run classic MOEAD_DL without DifferentialEvolution")
    p.add_argument('--use-gpu', action='store_true', default=True, help='Enable GPU')
    p.add_argument('--n_generations', type=int, default=25, help='Generations')
    p.add_argument('--h_divisions', type=int, default=49, help='H divisions (aumentado para espacio expandido)')
    p.add_argument('--n_neighbors', type=int, default=10, help='Neighbors')
    p.add_argument('--patience', type=int, default=5, help='Epochs de paciencia para early stopping en entrenamiento de')
    p.add_argument('--n_r', type=int, default=2, help='Max replacements')
    p.add_argument('--organo', type=str, default='ctv', help='Órgano a entrenar, usado en nombres de archivo y resultados')
    p.add_argument('--log', type=str, default='classic_moead_dl_log_ctv.json', help='Log file')
    p.add_argument('--checkpoint', type=str, default='classic_moead_dl_checkpoint_ctv.pkl', help='Checkpoint')
    p.add_argument('--output-metadata', type=str, default='classic_moead_dl_metadata.json', help='Metadata JSON output file')
    return p.parse_args()


def configure_device(use_gpu: bool):
    if use_gpu:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        try:
            import tensorflow as tf

            # ---------------------------------------------------------
            # PARCHE ESTRUCTURAL: DESACTIVAR LAYOUT OPTIMIZER
            # Previene el colapso (INVALID_ARGUMENT: Size of values 0)
            # al evaluar arquitecturas dinámicas con Dropout y sin sesgo.
            # ---------------------------------------------------------
            tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

            # Limitar la VRAM para que TF no la acapare toda de golpe
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    print(f"Advertencia al configurar VRAM: {e}")
                    
        except ImportError:
            pass
        print("--> GPU enabled (Layout Optimizer Desactivado para Estabilidad)")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("--> GPU disabled, using CPU")


def load_data(root_path: Path, organo: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = root_path / 'data'
    path_x = data_dir / f'X_train_{organo}_5k.npy'
    path_y = data_dir / f'Y_train_{organo}_5k.npy'

    print(f"--> Cargando datos desde: {data_dir}")

    if not path_x.exists() or not path_y.exists():
        raise FileNotFoundError(f"No se encontraron los archivos .npy en {data_dir}")

    try:
        X = np.load(path_x).astype(np.float32)
        Y = np.load(path_y).astype(np.float32)
        print(f"--> Datos cargados y convertidos a float32. Shape X: {X.shape}")
    except Exception as e:
        raise RuntimeError(f"Error cargando .npy: {e}")

    X_t, X_v, Y_t, Y_v = train_test_split(X, Y, test_size=0.2, random_state=42)
    del X, Y
    return X_t, Y_t, X_v, Y_v


def plot_front(archive, out_path: Path):
    if not archive:
        return
    f1 = [s.objectives[0] for s in archive]
    f2 = [s.objectives[1] for s in archive]

    plt.figure(figsize=(8, 6))
    plt.scatter(f1, f2, s=20, c='blue', alpha=0.6)
    plt.xlabel('Dice Loss (Minimizar)')
    plt.ylabel('Norm Params (Minimizar)')
    plt.title(f'MOEAD-DL Clásico Pareto Front ({len(archive)} soluciones)')
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

        if len(archive_sizes) == 0:
            return

        gens = np.arange(len(archive_sizes))
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        if z_star.shape[0] > 0 and z_star.shape[1] >= 2:
            plt.plot(gens, z_star[:, 0], label='Best Dice Loss')
            plt.plot(gens, z_star[:, 1], label='Best Size')
        plt.title('Evolución del Punto Ideal (Z*)')
        plt.legend(); plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(gens, archive_sizes, 'r-o', label='Archive Size')
        plt.xlabel('Generación')
        plt.title('Tamaño del Archivo Externo')
        plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Error ploteando historial: {e}")


def solution_summary(solution) -> dict[str, object]:
    return {
        'objectives': np.asarray(solution.objectives).tolist() if hasattr(solution, 'objectives') else None,
        'constraints': np.asarray(solution.constraints).tolist() if hasattr(solution, 'constraints') else None,
        'variables': np.asarray(solution.variables).tolist() if hasattr(solution, 'variables') else None,
        'model_config': getattr(solution, 'model_config', None),
    }


def save_metadata(
    output_path: Path,
    start_time: float,
    end_time: float,
    moead: MOEAD_DL,
    scalarization: Any,
    evo_op: Any,
    organo: str,
    metadata_file: Path,
):
    metadata = {
        'algorithm': 'MOEAD_DL',
        'evolutionary_operator': evo_op.__class__.__name__,
        'scalarization': scalarization.__class__.__name__,
        'organo': organo,
        'n_generations': moead.n_gen,
        'h_divisions': moead.h_divisions,
        'n_neighbors': moead.n_neighbors,
        'n_r': moead.n_r,
        'population_size': moead.n_pop,
        'elapsed_seconds': end_time - start_time,
        'start_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(start_time)),
        'end_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(end_time)),
        'population': [solution_summary(sol) for sol in moead.population],
        'neighborhoods': moead.neighborhoods.tolist() if hasattr(moead, 'neighborhoods') else None,
        'log_file': str(output_path / metadata_file) if output_path else None,
    }

    with open(output_path / metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    configure_device(args.use_gpu)

    try:
        X_train, Y_train, X_val, Y_val = load_data(PROJECT_ROOT, args.organo)
    except Exception as e:
        print(f"FATAL: {e}")
        return

    problem = DLProblem(X_train, Y_train, X_val, Y_val, batch_size=8,
                        patience=args.patience)
    
    scalarization = PBI()
    evo_op = CrossoverMutation(
        crossover=SBXCrossover(eta=15.0, prob_cross=0.9),
        mutation=PolynomialMutation(eta=20.0, prob_mut=None),
        mating_prob=0.9,
    )

    output_dir = PROJECT_ROOT / f'resultados' / f'resultado_{args.organo}'
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / args.log
    checkpoint_path = output_dir / args.checkpoint
    metadata_path = Path(args.output_metadata)

    print(f"--> Iniciando MOEAD_DL clásico: Gens={args.n_generations}, Vecinos={args.n_neighbors}, Órgano={args.organo}")
    print(f"--> Guardando resultados en: {output_dir}")

    moead = MOEAD_DL(
        problem=problem,
        scalarization=scalarization,
        evolutionary_op=evo_op,
        h_divisions=args.h_divisions,
        n_neighbors=args.n_neighbors,
        n_generations=args.n_generations,
        n_r=args.n_r,
        log_filename=str(log_path),
        checkpoint_file=str(checkpoint_path),
    )

    start_time = time.time()
    archive, history = moead.run()
    end_time = time.time()

    plot_front(archive, output_dir / 'classic_moead_dl_front.png')
    plot_history(history, output_dir / 'classic_moead_dl_history.png')
    save_metadata(output_dir, start_time, end_time, moead, scalarization, evo_op, args.organo, metadata_path)

    print(f"--> Ejecución finalizada. Metadata guardada en {output_dir / metadata_path}")


if __name__ == '__main__':
    main()

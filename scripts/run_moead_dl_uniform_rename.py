"""
Runner Discreto de MOEAD_DL optimizado para NAS.

Este script utiliza la reproducción de MOEA/D basada en Cruce Uniforme 
(UniformCrossover) y Mutación Uniforme Acotada, operando bajo la 
escalarización de Tchebycheff con corrección asintótica para espacios discretos.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Importaciones ajustadas a la nueva arquitectura geométrica discreta
from moead.algorithms import MOEAD_DL
from moead.crossovers import UniformCrossover
# Asegúrate de ajustar la ruta de importación de tu mutación acotada según tu directorio
from moead.mutations import BoundedUniformMutation 
from moead.evolutionary_operator import CrossoverMutation
from moead.scalarizations import Tchebycheff
from moead.problems import DLProblemRefactor


def parse_args():
    p = argparse.ArgumentParser(description="Run MOEAD_DL with Discrete Uniform Operators for NAS")
    p.add_argument('--use-gpu', action='store_true', default=True, help='Enable GPU')
    p.add_argument('--timeout-per-evaluation', type=int, default=1800, help='Timeout per evaluation in seconds')
    p.add_argument('--n_generations', type=int, default=25, help='Generations')
    p.add_argument('--h_divisions', type=int, default=49, help='H divisions para vectores lambda')
    p.add_argument('--n_neighbors', type=int, default=10, help='Neighbors')
    p.add_argument('--patience', type=int, default=5, help='Paciencia para early stopping')
    p.add_argument('--epochs', type=int, default=20, help='Número máximo de epochs por arquitectura')
    p.add_argument('--batch_size', type=int, default=8, help='Batch size físico para entrenamiento (VRAM control)')
    p.add_argument('--n_r', type=int, default=2, help='Max replacements en vecindario')
    p.add_argument('--organo', type=str, default='ctv', help='Órgano objetivo de segmentación')
    p.add_argument('--log', type=str, default='uniform_moead_dl_log_ctv.json', help='Log file')
    p.add_argument('--checkpoint', type=str, default='uniform_moead_dl_checkpoint_ctv.pkl', help='Checkpoint')
    p.add_argument('--output-metadata', type=str, default='uniform_moead_dl_metadata.json', help='Metadata JSON output file')
    p.add_argument('--verbose', type=int, default=1, help='0: Silencio | 1: Info Clave | 2: Debug (I/O Keras activo)')
    
    return p.parse_args()

def plot_front(archive, out_path: Path):
    if not archive:
        return
    f1 = [s.objectives[0] for s in archive]
    f2 = [s.objectives[1] for s in archive]

    plt.figure(figsize=(8, 6))
    plt.scatter(f1, f2, s=20, c='blue', alpha=0.6)
    plt.xlabel('Dice Loss (Minimizar)')
    plt.ylabel('Norm Params (Minimizar)')
    plt.title(f'MOEAD-DL Pareto Front Discreto ({len(archive)} soluciones)')
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
        plt.title('Tamaño del Archivo Externo de Pareto')
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
        'crossover_type': evo_op.crossover_op.__class__.__name__,
        'mutation_type': evo_op.mutation_op.__class__.__name__,
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
    print(f"--> Timeout por evaluación: {args.timeout_per_evaluation} segundos")
    output_dir = PROJECT_ROOT / 'resultados' / f'resultado_discreto_{args.organo}'
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / args.log
    checkpoint_path = output_dir / args.checkpoint
    metadata_path = Path(args.output_metadata)

    cache_json_path = output_dir / f'nas_evaluation_cache_{args.organo}.json'

    data_dir = PROJECT_ROOT / 'data'
    path_x = data_dir / f'X_train_{args.organo}_5k.npy'
    path_y = data_dir / f'Y_train_{args.organo}_5k.npy'
    # 1. Instanciación del entorno evaluativo (Fenotipo)
    problem = DLProblemRefactor(X_data=path_x, Y_data=path_y,
                                train_batch_size=args.batch_size,
                                gradient_accumulation_steps=2,
                                val_batch_size=args.batch_size,
                                epochs=args.epochs,
                                patience=args.patience,
                                verbose=args.verbose,
                                cache_path=cache_json_path,
                                timeout_per_evaluation=args.timeout_per_evaluation,
                                use_gpu=args.use_gpu)
    
    # 2. Heurística de la Tasa de Mutación
    # Se extrae dinámicamente el número de variables de decisión
    n_variables = len(problem.bounds)
    dynamic_mutation_rate = 1.0 / float(n_variables)
    
    # 3. Ensamblaje de Operadores Discretos
    # Tchebycheff asegura contornos ortogonales ideales para el cruce uniforme
    scalarization = Tchebycheff(epsilon=1e-6)
    
    evo_op = CrossoverMutation(
        crossover=UniformCrossover(prob_cross=0.9),
        mutation=BoundedUniformMutation(prob_mut=dynamic_mutation_rate),
        mating_prob=0.9,
    )

    print(f"\n--> Iniciando MOEAD_DL Discreto (NAS):")
    print(f"    - Generaciones: {args.n_generations} | Vecinos: {args.n_neighbors} | Órgano: {args.organo}")
    print(f"    - Cruce: UniformCrossover | Mutación: BoundedUniformMutation (Tasa: {dynamic_mutation_rate:.3f})")
    print(f"    - Guardando resultados en: {output_dir}\n")

    # 4. Orquestación del Motor Evolutivo
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

    # 5. Generación de Auditoría y Reportes Geométricos
    plot_front(archive, output_dir / 'uniform_moead_dl_front.png')
    plot_history(history, output_dir / 'uniform_moead_dl_history.png')
    save_metadata(output_dir, start_time, end_time, moead, scalarization, evo_op, args.organo, metadata_path)

    print(f"--> Ejecución finalizada con éxito. Metadata guardada en {output_dir / metadata_path}")


if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
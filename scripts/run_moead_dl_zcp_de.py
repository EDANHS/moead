"""
Runner Vectorial de MOEAD_ZCP optimizado para NAS (Training-Free).

Este script orquesta el algoritmo MOEA/D utilizando la reproducción basada en Evolución 
Diferencial (DifferentialEvolution) acoplada a un filtro subrogado (ZCPMoveProposal).
La inicialización poblacional es acelerada mediante Zero-Cost Proxies (Warmup) escalables,
permitiendo la optimización multiobjetivo en el hiperespacio topológico 
con un coste computacional O(1) y erradicando el cuello de botella de la GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Configuración dinámica del entorno y rutas maestras
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==============================================================================
# IMPORTACIONES ACOPLADAS A LA NUEVA ARQUITECTURA ANALÍTICA (ZCP)
# ==============================================================================
from moead.scalarizations import PBI
from moead.evolutionary_operator import DifferentialEvolution

# Módulos del Ecosistema Zero-Cost
from moead.algorithms import MOEAD_ZCP
from moead.problems import DLProblemZCP
from moead.utils import SurrogatePredictor
from moead.initializer import ZeroCostWarmup
from moead.evolutionary_operator import ZCPMoveProposal


def parse_args():
    p = argparse.ArgumentParser(description="Ejecución de MOEAD_ZCP con Evolución Diferencial y PBI para NAS")
    p.add_argument('--use-gpu', action='store_true', default=True, help='Habilita el uso de la GPU (Limitado a extracción ZCP)')
    p.add_argument('--timeout-per-evaluation', type=int, default=90, help='Timeout por evaluación (reducido gracias a ZCP)')
    p.add_argument('--n_generations', type=int, default=30, help='Número total de generaciones a simular')
    p.add_argument('--h_divisions', type=int, default=49, help='H divisions para los vectores lambda')
    p.add_argument('--n_neighbors', type=int, default=10, help='Tamaño del vecindario (T)')
    p.add_argument('--n_r', type=int, default=2, help='Máximos reemplazos permitidos por subproblema')
    p.add_argument('--organo', type=str, default='ctv', help='Órgano objetivo para la red')
    
    # [NUEVOS ARGUMENTOS] Aislamiento de Experimentos y Arquitectura ZCP
    p.add_argument('--experiment-name', type=str, default='zcp_de_baseline', help='Directorio de salida para aislar los resultados de esta ejecución')
    p.add_argument('--surrogate-model', type=str, default='rf_surrogate_model.joblib', help='Ruta al binario del modelo Random Forest pre-entrenado')
    p.add_argument('--warmup-multiplier', type=int, default=10, help='Factor multiplicador escalar poblacional para el Warmup ZCP')
    p.add_argument('--pool-size', type=int, default=5, help='Descendientes generados y filtrados por el Move Proposal (DE)')
    
    # Nomenclaturas adaptadas al marco híbrido
    p.add_argument('--log', type=str, default='zcp_de_moead_log.json', help='Log de transacciones evolutivas')
    p.add_argument('--checkpoint', type=str, default='zcp_de_moead_checkpoint.pkl', help='Punto de restauración de memoria')
    p.add_argument('--output-metadata', type=str, default='zcp_de_moead_metadata.json', help='Metadatos y telemetría de ejecución')
    p.add_argument('--verbose', type=int, default=1, help='0: Silencio | 1: Info Clave | 2: Trazabilidad profunda')
    
    return p.parse_args()


def plot_front(archive, out_path: Path):
    if not archive:
        return
    f1 = [s.objectives[0] for s in archive]
    f2 = [s.objectives[1] for s in archive]

    plt.figure(figsize=(8, 6))
    # Paleta visual ajustada a tonos azules para diferenciar el paradigma ZCP de las ejecuciones empíricas
    plt.scatter(f1, f2, s=25, c='dodgerblue', alpha=0.7, edgecolors='darkblue') 
    plt.xlabel('Dice Loss Estimado (Minimizar)')
    plt.ylabel('Complejidad Paramétrica Normalizada (Minimizar)')
    plt.title(f'MOEAD-ZCP Pareto Front Vectorial (DE+PBI) ({len(archive)} soluciones)')
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
            plt.plot(gens, z_star[:, 0], label='Est. Best Dice Loss (Oráculo)', color='navy', linewidth=2)
            plt.plot(gens, z_star[:, 1], label='Best Params Norm', color='cyan', linewidth=2)
        plt.title('Evolución del Punto Ideal (Z*) - Inferencia Subrogada')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(2, 1, 2)
        plt.plot(gens, archive_sizes, color='teal', marker='o', label='Archive Size')
        plt.xlabel('Generación Evolutiva')
        plt.title('Dinámica de Crecimiento del Archivo Externo de Pareto')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Error ploteando historial topológico: {e}")


def solution_summary(solution) -> dict[str, object]:
    return {
        'objectives': np.asarray(solution.objectives).tolist() if hasattr(solution, 'objectives') else None,
        'constraints': np.asarray(solution.constraints).tolist() if hasattr(solution, 'constraints') else None,
        'variables': np.asarray(solution.variables).tolist() if hasattr(solution, 'variables') else None,
        'model_config': getattr(solution, 'model_config', None),
    }


def save_metadata(
    output_path: Path, start_time: float, end_time: float,
    moead: MOEAD_ZCP, scalarization: Any, evo_op: Any,
    args: argparse.Namespace, metadata_file: Path
):
    # Cálculo inverso de la población real para persistencia de contexto
    n_pop_estimada = args.h_divisions + 1
    
    metadata = {
        'algorithm': 'MOEAD_ZCP',
        'evolutionary_operator': 'ZCPMoveProposal_DE',
        'scalarization': scalarization.__class__.__name__,
        'organo': args.organo,
        'zcp_configuration': {
            'warmup_multiplier': args.warmup_multiplier,
            'calculated_warmup_size': n_pop_estimada * args.warmup_multiplier,
            'mutation_pool_size': args.pool_size,
            'surrogate_model_binary': args.surrogate_model
        },
        'n_generations': moead.n_gen,
        'h_divisions': moead.h_divisions,
        'n_neighbors': moead.n_neighbors,
        'n_r': moead.n_r,
        'population_size': moead.n_pop,
        'elapsed_seconds': end_time - start_time,
        'start_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(start_time)),
        'end_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(end_time)),
        'population': [solution_summary(sol) for sol in moead.population],
        'log_file': str(output_path / metadata_file) if output_path else None,
    }

    with open(output_path / metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    
    # Cálculo paramétrico adaptativo de la dimensión estructural
    n_pop_estimada = args.h_divisions + 1
    warmup_size_dinamico = n_pop_estimada * args.warmup_multiplier

    print("===================================================================")
    print(f"--> Iniciando Entorno NAS Acelerado (Training-Free)")
    print(f"--> Experimento: {args.experiment_name}")
    print(f"--> Órgano Objetivo: {args.organo.upper()}")
    print("===================================================================\n")
    
    # Aislamiento de persistencia de datos orientada a la experimentación ZCP + DE
    output_dir = PROJECT_ROOT / 'resultados' / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / args.log
    checkpoint_path = output_dir / args.checkpoint
    metadata_path = Path(args.output_metadata)

    # 1. Hidratación del Oráculo Predictivo
    surrogate_path = str(PROJECT_ROOT / args.surrogate_model)
    if not os.path.exists(surrogate_path):
        print(f"[CRÍTICO] No se localizó el binario del modelo predictivo en: {surrogate_path}")
        print("Por favor, asegúrese de haber ejecutado 'train_surrogate.py' previamente para generar el oráculo (.joblib).")
        sys.exit(1)
        
    surrogate = SurrogatePredictor(model_path=surrogate_path)
    
    # 2. Instanciación del Entorno Híbrido (DLProblemZCP)
    data_dir = PROJECT_ROOT / 'data'
    path_x = data_dir / f'X_train_{args.organo}_5k.npy'
    path_y = data_dir / f'Y_train_{args.organo}_5k.npy'
    
    problem = DLProblemZCP(
        surrogate_model=surrogate,
        X_data=path_x, Y_data=path_y,
        # Parámetros heredados (aunque no desencadenan Keras, se inyectan por polimorfismo estructural)
        train_batch_size=8, gradient_accumulation_steps=2, val_batch_size=8,
        epochs=1, patience=1, verbose=args.verbose, 
        cache_path=output_dir / f'nas_evaluation_cache_zcp_{args.organo}.json',
        timeout_per_evaluation=args.timeout_per_evaluation,
        use_gpu=args.use_gpu
    )
    
    scalarization = PBI()
    
    # 3. Construcción del Decorador Evolutivo (Descendencia Guiada)
    de_base = DifferentialEvolution()
    evo_op = ZCPMoveProposal(
        base_operator=de_base,
        surrogate_model=surrogate,
        pool_size=args.pool_size
    )

    print(f"--> Despliegue de Componentes Completado:")
    print(f"    • Población Objetivo M=2 (N = {n_pop_estimada} individuos)")
    print(f"    • Estrategia Inicialización: ZeroCostWarmup (Prospección = {warmup_size_dinamico} arquitecturas)")
    print(f"    • Operador Evolutivo: ZCPMoveProposal envolviendo Evolución Diferencial (Pool={args.pool_size})")
    print(f"    • Generaciones: {args.n_generations} | Vecinos: {args.n_neighbors}")
    print(f"    • Destino de los reportes: {output_dir}\n")

    # 4. Orquestación del Motor Multiobjetivo
    warmup_strategy = ZeroCostWarmup(warmup_size=warmup_size_dinamico)
    
    moead = MOEAD_ZCP(
        warmup_initializer=warmup_strategy,
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

    # 5. Generación de Auditoría, Matrices Geométricas y Metadatos
    plot_front(archive, output_dir / 'zcp_de_moead_dl_front.png')
    plot_history(history, output_dir / 'zcp_de_moead_dl_history.png')
    save_metadata(output_dir, start_time, end_time, moead, scalarization, evo_op, args, metadata_path)

    print(f"\n===================================================================")
    print(f"--> Trazabilidad finalizada con éxito en {end_time - start_time:.2f} segundos.")
    print(f"--> Telemetría evolutiva consolidada en {output_dir / metadata_path}")
    print("===================================================================")


if __name__ == '__main__':
    import multiprocessing
    try:
        # Obligatorio para asegurar el aislamiento de las llamadas de inferencia de Keras
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
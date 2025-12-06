import os
import tempfile
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from moead.algorithms import MOEAD_DL
from moead.scalarizations import WeightedSum
from moead.evolutionary_operator import DifferentialEvolution
from moead.problems import DLProblem
from moead.solutions import DLSolution


IMG_DIR = ".images"
os.makedirs(IMG_DIR, exist_ok=True)


def plot_front_dl(pareto_front: list, save_path: str, title: str = "MOEAD-DL Front"):
    # Pareto front: objective 0 = dice_loss (0..1), objective 1 = n_params (int, large)
    f1 = [s.objectives[0] for s in pareto_front]
    f2 = [s.objectives[1] for s in pareto_front]

    plt.figure(figsize=(8, 6))
    plt.scatter(f1, f2, s=20, c='blue', alpha=0.7)
    plt.yscale('log')  # params can be large; usar escala log para visualización
    plt.xlabel('Dice Loss (minimize)')
    plt.ylabel('Number of Params (log scale)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_history(history_log: dict, save_path: str, title: str = "MOEAD-DL History"):
    try:
        z_star_history = history_log['z_star_per_gen']
        archive_size_history = history_log['archive_size_per_gen']
    except Exception:
        return

    if not z_star_history:
        return

    z_arr = np.array(z_star_history)
    gen = np.arange(len(archive_size_history))

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(gen, z_arr[:, 0], 'b-', label='z*_1 (Dice)')
    plt.plot(gen, z_arr[:, 1], 'g-', label='z*_2 (Params)')
    plt.ylabel('z* values')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(gen, archive_size_history, 'r-o', label='Archive size')
    plt.xlabel('Generation')
    plt.ylabel('Archive size')
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def test_moead_dl_generates_front_and_logs_model_configs():
    # Datos sintéticos mínimos (DLProblem los acepta pero su evaluación es simulada)
    X_train = np.zeros((1, 1))
    Y_train = np.zeros((1, 1))
    X_val = np.zeros((1, 1))
    Y_val = np.zeros((1, 1))

    problem = DLProblem(X_train, Y_train, X_val, Y_val)

    scalarization = WeightedSum()
    de_op = DifferentialEvolution(F=0.5, CR=0.9, selection_prob=0.9)

    # Usamos archivos temporales para logger/checkpoint para no ensuciar el repo
    checkpoint_file = "moead_dl_checkpoint.txt"
    history_log = "moead_dl_log.json"
    
    try:
        moead = MOEAD_DL(
            problem=problem,
            scalarization=scalarization,
            evolutionary_op=de_op,
            h_divisions=50,      # Genera población pequeña (m=2 -> n_pop = h+1 = 3)
            n_neighbors=20,
            n_generations=10000,    # Ejecutar 1 generación para rapidez
            n_r=2,
            log_filename=history_log,
            checkpoint_file=checkpoint_file
        )

        archive, history = moead.run()

        # Comprobaciones básicas
        assert isinstance(archive, list)
        assert isinstance(history, dict)

        # Plot front image
        front_path = os.path.join(IMG_DIR, "moead_dl_front.png")
        plot_front_dl(archive, front_path, title="MOEAD-DL Pareto Front")
        assert os.path.exists(front_path)

        # Plot history
        history_path = os.path.join(IMG_DIR, "moead_dl_history.png")
        plot_history(history, history_path, title="MOEAD-DL History")
        assert os.path.exists(history_path)

        # Leer JSONLogger y comprobar que las configuraciones de los modelos están presentes
        with open(history_log, 'r') as f:
            log_data = json.load(f)

        gens = log_data.get('metadata', {}).get('generations_data', [])
        assert len(gens) >= 1

        # Extraer configuraciones registradas y guardarlas en un archivo para inspección
        configs_path = os.path.join(IMG_DIR, "moead_dl_model_configs.txt")
        with open(configs_path, 'w', encoding='utf-8') as cf:
            for g in gens:
                cf.write(f"Generation {g.get('generation')} (archive_size={g.get('archive_size')})\n")
                for sol in g.get('solutions', []):
                    cfg = sol.get('model_config')
                    cf.write(json.dumps(cfg, ensure_ascii=False) + "\n")
                cf.write("\n")

        assert os.path.exists(configs_path)
    except Exception as e:
        assert False, f"Test failed with exception: {e}"

if __name__ == "__main__":
    # Ejecuta la prueba principal de MOEAD-DL cuando se llama como script
    test_moead_dl_generates_front_and_logs_model_configs()
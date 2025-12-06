import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from moead.crossovers import SBXCrossover
from moead.mutations import PolynomialMutation
from moead.algorithms import MOEAD
from moead.problems import Problem, ZDT6Problem
from moead.scalarizations import PBI, Tchebycheff
from moead.evolutionary_operator import CrossoverMutation, DifferentialEvolution


IMG_DIR = ".images"
os.makedirs(IMG_DIR, exist_ok=True)

def test_zdt6_run_with_sbx():
    """
    Prueba ZDT6 con los operadores estándar (SBX + Polinómica)
    y la selección de padres híbrida.
    """
    print("\nIniciando test ZDT6 (con SBX + PolyMut)...")
    
    # 1. Configurar las "Estrategias"
    problem = ZDT6Problem(n_vars=10) # <-- CAMBIO
    scalarization = Tchebycheff()
    
    crossover_op = SBXCrossover(eta=15, prob_cross=0.9)
    mutation_op = PolynomialMutation(eta=20)
    
    evo_op = CrossoverMutation(
        crossover=crossover_op,
        mutation=mutation_op,
        mating_prob=0.9 # Estrategia híbrida
    )
    
    # 2. Inyectar dependencias en el solver
    moead_solver = MOEAD(
        problem=problem,
        scalarization=scalarization,
        evolutionary_op=evo_op,
        h_divisions=99,    # N=100
        n_neighbors=20,
        n_generations=200, # <-- CAMBIO (Más generaciones)
        n_r=2              # Límite de reemplazo
    )
    
    # 3. Ejecutar
    pareto_front, history_log = moead_solver.run()
    
    # 4a. Guardar imagen del Frente (Verificación Visual)
    save_path = os.path.join(IMG_DIR, "zdt6_front_sbx.png") # <-- CAMBIO
    plot_results_zdt6(pareto_front, problem, "ZDT6 con SBX", save_path) # <-- CAMBIO
    print(f"Test ZDT6 (SBX) completado. Imagen guardada en: {save_path}")

    # 4b. Guardar imagen del Historial (Análisis de Proceso)
    history_save_path = os.path.join(IMG_DIR, "zdt6_history_sbx.png") # <-- CAMBIO
    plot_history(history_log, "ZDT6 con SBX", history_save_path)
    print(f"Historial ZDT6 (SBX) guardado en: {history_save_path}")

    # 5. Verificar con Asserts (Verificación Automática)
    verify_results_zdt6(pareto_front, problem, avg_error_limit=0.1) # <-- CAMBIO


# ---
# PRUEBA 2: ZDT6 con Evolución Diferencial (DE)
# ---

def test_zdt6_run_with_de():
    """
    Prueba ZDT6 con la estrategia de Evolución Diferencial (MOEA/D-DE).
    """
    print("\nIniciando test ZDT6 (con Evolución Diferencial)...")
    
    # 1. Configurar las "Estrategias"
    problem = ZDT6Problem(n_vars=10) # <-- CAMBIO
    scalarization = Tchebycheff()
    evo_op = DifferentialEvolution(F=0.5, CR=0.6) # Estrategia DE
    
    # 2. Inyectar dependencias en el solver
    moead_solver = MOEAD(
        problem=problem,
        scalarization=scalarization,
        evolutionary_op=evo_op,
        h_divisions=99,    # N=100
        n_neighbors=20,
        n_generations=200, # <-- CAMBIO (Más generaciones)
        n_r=2             # Con DE, n_r puede ser más grande (ej. T)
    )
    
    # 3. Ejecutar
    pareto_front, history_log = moead_solver.run()
    
    # 4a. Guardar imagen del Frente (Verificación Visual)
    save_path = os.path.join(IMG_DIR, "zdt6_front_de.png") # <-- CAMBIO
    plot_results_zdt6(pareto_front, problem, "ZDT6 con DE", save_path) # <-- CAMBIO
    print(f"Test ZDT6 (DE) completado. Imagen guardada en: {save_path}")

    # 4b. Guardar imagen del Historial (Análisis de Proceso)
    history_save_path = os.path.join(IMG_DIR, "zdt6_history_de.png") # <-- CAMBIO
    plot_history(history_log, "ZDT6 con DE", history_save_path)
    print(f"Historial ZDT6 (DE) guardado en: {history_save_path}")

    # 5. Verificar con Asserts (Verificación Automática)
    verify_results_zdt6(pareto_front, problem, avg_error_limit=0.05) # <-- CAMBIO (DE es mejor)


# ---
# FUNCIONES DE AYUDA (Específicas para ZDT6)
# ---

def plot_results_zdt6(pareto_front: list, problem: Problem, title: str, save_path: str):
    """Función genérica para graficar los resultados de ZDT6."""
    
    f1_alg = [sol.objectives[0] for sol in pareto_front]
    f2_alg = [sol.objectives[1] for sol in pareto_front]

    # Generar el frente de Pareto real (Fórmula de ZDT6)
    # f1 = 1 - exp(-4*x1) * sin^6(6*pi*x1)
    # f2 = 1 - f1^2
    x1_real = np.linspace(0, 1, 1000)
    f1_real = 1.0 - np.exp(-4.0 * x1_real) * (np.sin(6.0 * np.pi * x1_real)**6)
    f2_real = 1 - f1_real**2
    
    # Ordenar los puntos para un ploteo de línea correcto
    sort_indices = np.argsort(f1_real)
    f1_real_sorted = f1_real[sort_indices]
    f2_real_sorted = f2_real[sort_indices]

    plt.figure(figsize=(10, 7))
    plt.scatter(f1_alg, f2_alg, s=10, c='blue', label='Resultado (del Archive)', alpha=0.6, zorder=2)
    plt.plot(f1_real_sorted, f2_real_sorted, 'r-', linewidth=2.5, label='Frente Real ZDT6', zorder=3) # <-- CAMBIO
    plt.title(f'Verificación del Frente de Pareto ({title})')
    plt.xlabel('Objetivo 1 (f1)')
    plt.ylabel('Objetivo 2 (f2)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=-0.1, top=1.5)

    plt.savefig(save_path)
    plt.close()

def plot_history(history_log: dict, title: str, save_path: str):
    """
    Grafica el historial de convergencia (z_star) y el 
    crecimiento del archive a lo largo de las generaciones.
    """
    # (Esta función es genérica y no necesita cambios)
    try:
        z_star_history = history_log['z_star_per_gen']
        archive_size_history = history_log['archive_size_per_gen']
        generations = np.arange(len(archive_size_history))
    except KeyError:
        print("Error: El 'history_log' no tiene las claves esperadas.")
        return
    except TypeError:
        print(f"Error: 'history_log' no es un diccionario: {history_log}")
        return

    if not z_star_history:
        print("Error: El historial de z_star está vacío.")
        return

    z_star_transposed = np.array(z_star_history).T
    f1_ideal_history = z_star_transposed[0]
    f2_ideal_history = z_star_transposed[1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(generations, f1_ideal_history, 'b-', label='Ideal f1 (z*_1)')
    ax1.plot(generations, f2_ideal_history, 'g-', label='Ideal f2 (z*_2)')
    ax1.set_title(f'Historial de Convergencia ({title})')
    ax1.set_ylabel('Valor Objetivo (Ideal)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(generations, archive_size_history, 'r-', label='Tamaño del Archive')
    ax2.set_title('Historial de Diversidad (Archive)')
    ax2.set_xlabel('Generación')
    ax2.set_ylabel('Nº de Soluciones (Archive)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def verify_results_zdt6(pareto_front: list, problem: Problem, avg_error_limit: float):
    """Función genérica para verificar los resultados de ZDT6 con asserts."""
    
    print("Verificando resultados...")
    
    if len(pareto_front) == 0:
        assert False, "¡El test falló! El frente de Pareto está vacío."

    # 1. Test de Convergencia (Error Promedio)
    total_error = 0
    for sol in pareto_front:
        f1 = sol.objectives[0]
        f2_actual = sol.objectives[1]
        
        f1_safe = max(f1, 1e-6)
            
        # Fórmula de ZDT6 (frente real)
        f2_ideal = 1 - f1_safe**2 # <-- CAMBIO
        total_error += abs(f2_actual - f2_ideal)
        
    avg_error = total_error / len(pareto_front)
    
    convergencia_ok = avg_error < avg_error_limit
    print(f"  Error de convergencia promedio: {avg_error:.4f} (Límite: {avg_error_limit})")
    assert convergencia_ok, f"¡El test falló! Error de convergencia muy alto: {avg_error}"

    # 2. Test de Diversidad (Cobertura del Frente)
    f1_values = np.array([sol.objectives[0] for sol in pareto_front])
    min_f1 = np.min(f1_values)
    max_f1 = np.max(f1_values)
    
    # El frente de ZDT6 no empieza en f1=0. Comienza alrededor de f1=0.28
    diversidad_min_ok = min_f1 < 0.35 # <-- CAMBIO
    diversidad_max_ok = max_f1 > 0.95 # <-- CAMBIO
    
    print(f"  Cobertura del frente (f1): min={min_f1:.4f} (Req: < 0.35), max={max_f1:.4f} (Req: > 0.95)")
    assert diversidad_min_ok, f"¡El test falló! No se encontraron soluciones cerca del inicio (min f1: {min_f1})"
    assert diversidad_max_ok, f"¡El test falló! No se encontraron soluciones cerca del final (max f1: {max_f1})"

    print("¡Verificación completada! El test pasó.")


if __name__ == "__main__":
    op = 3
    if op == 1:
        test_zdt6_run_with_sbx()
    elif op == 2:
        test_zdt6_run_with_de()
    else:
        test_zdt6_run_with_sbx()
        test_zdt6_run_with_de()
        
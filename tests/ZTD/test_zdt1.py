import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend estable
import matplotlib.pyplot as plt

# --- IMPORTACIÓN DE DEPENDENCIAS ---
from moead.crossovers import SBXCrossover
from moead.mutations import PolynomialMutation
from moead.algorithms import MOEAD
from moead.problems import Problem, ZDT1Problem
from moead.scalarizations import PBI, Tchebycheff
from moead.evolutionary_operator import CrossoverMutation, DifferentialEvolution

# --- CONFIGURACIÓN DEL ENTORNO ---
IMG_DIR = "seminar_results_zdt1"
os.makedirs(IMG_DIR, exist_ok=True)

# ==============================================================================
# BLOQUE 1: MOTORES DE RENDERIZADO (VISUALIZACIÓN INDIVIDUAL)
# ==============================================================================

def plot_pareto_front(pareto_front, title, filename):
    """
    Genera la validación geométrica para ZDT1.
    El frente es CONVEXO (f2 = 1 - sqrt(f1)).
    """
    save_path = os.path.join(IMG_DIR, filename)
    
    # Extraer objetivos
    f1_alg = [sol.objectives[0] for sol in pareto_front]
    f2_alg = [sol.objectives[1] for sol in pareto_front]

    # Generar Frente Real Teórico (ZDT1)
    x1_real = np.linspace(0, 1, 500)
    f1_real = x1_real
    f2_real = 1 - np.sqrt(f1_real) # Fórmula Convexa
    
    plt.figure(figsize=(10, 7))
    
    # Capa 1: Soluciones del Algoritmo
    plt.scatter(f1_alg, f2_alg, s=20, c='blue', alpha=0.6, label='Solución Obtenida', zorder=2)
    
    # Capa 2: Frente Teórico
    plt.plot(f1_real, f2_real, 'r-', linewidth=2.5, label='Frente Teórico (ZDT1)', zorder=1)
    
    plt.title(f'Validación de Convergencia (Convexo): {title}')
    plt.xlabel('Objetivo 1 (Minimizar)')
    plt.ylabel('Objetivo 2 (Minimizar)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.1)
    
    plt.savefig(save_path)
    plt.close()
    print(f"   [Gráfico] Frente de Pareto guardado en: {save_path}")

def plot_convergence_history(history_log, title, filename):
    """
    Visualiza la dinámica de convergencia.
    """
    save_path = os.path.join(IMG_DIR, filename)
    
    if not history_log or 'z_star_per_gen' not in history_log:
        return

    z_star_history = np.array(history_log['z_star_per_gen']).T
    archive_history = history_log['archive_size_per_gen']
    generations = np.arange(len(archive_history))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Convergencia Z*
    ax1.plot(generations, z_star_history[0], 'b-', label='Z*_1')
    ax1.plot(generations, z_star_history[1], 'g-', label='Z*_2')
    ax1.set_title(f'Dinámica de Convergencia: {title}')
    ax1.set_ylabel('Punto Ideal')
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    
    # Diversidad (Archive)
    ax2.plot(generations, archive_history, 'r-', label='Tamaño Archive')
    ax2.set_title('Evolución de la Población No Dominada')
    ax2.set_xlabel('Generaciones')
    ax2.set_ylabel('Nº Soluciones')
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   [Gráfico] Historial guardado en: {save_path}")

# ==============================================================================
# BLOQUE 2: LÓGICA DE NEGOCIO (MÉTRICAS Y EJECUCIÓN)
# ==============================================================================

def calculate_metrics_zdt1(pareto_front):
    """
    KPIs específicos para ZDT1.
    """
    if not pareto_front:
        return 1.0, 0.0

    total_error = 0
    f1_vals = []
    
    for sol in pareto_front:
        f1, f2 = sol.objectives
        f1_vals.append(f1)
        
        # Frente ideal ZDT1: f2 = 1 - sqrt(f1)
        f1_safe = max(f1, 0.0)
        f2_ideal = 1 - np.sqrt(f1_safe)
        
        # Distancia vertical al frente
        total_error += abs(f2 - f2_ideal)
        
    avg_error = total_error / len(pareto_front)
    
    # Spread
    spread = max(f1_vals) - min(f1_vals) if f1_vals else 0
    
    return avg_error, spread

def run_experiment_zdt1(algorithm_type):
    """
    Orquestador para ZDT1.
    """
    problem = ZDT1Problem(n_vars=30) 
    
    if algorithm_type == 'SBX':
        print("\n--> Iniciando ZDT1 (Convexo) con MOEA/D Estándar (SBX)...")
        evo_op = CrossoverMutation(
            crossover=SBXCrossover(eta=15, prob_cross=0.9),
            mutation=PolynomialMutation(eta=20),
            mating_prob=0.9
        )
        label = "Estándar (SBX)"
        file_suffix = "sbx"
        
    elif algorithm_type == 'DE':
        print("\n--> Iniciando ZDT1 (Convexo) con MOEA/D Evolución Diferencial (DE)...")
        evo_op = DifferentialEvolution(F=0.5, CR=0.9) # CR alto para ZDT1 suele funcionar bien
        label = "Propuesto (DE)"
        file_suffix = "de"
    
    # Solver Config
    solver = MOEAD(
        problem=problem,
        scalarization=Tchebycheff(),
        evolutionary_op=evo_op,
        h_divisions=99,
        n_neighbors=20,
        n_generations=150, 
        n_r=2 
    )
    
    pareto_front, history = solver.run()
    
    # Visualización
    plot_pareto_front(pareto_front, f"ZDT1 - {label}", f"zdt1_front_{file_suffix}.png")
    plot_convergence_history(history, label, f"zdt1_history_{file_suffix}.png")
    
    # Métricas
    error, spread = calculate_metrics_zdt1(pareto_front)
    print(f"   [Métricas] {label} -> Error: {error:.5f} | Spread: {spread:.5f}")
    
    return error, spread

# ==============================================================================
# BLOQUE 3: SÍNTESIS COMPARATIVA (GENERADOR DE SLIDES)
# ==============================================================================

def generate_comparative_charts_zdt1(sbx_metrics, de_metrics):
    """
    Genera gráficos comparativos.
    """
    print("\n--> Generando gráficos comparativos de alto nivel para ZDT1...")
    
    strategies = ['MOEA/D-SBX\n(Estándar)', 'MOEA/D-DE\n(Propuesto)']
    errors = [sbx_metrics[0], de_metrics[0]]
    spreads = [sbx_metrics[1], de_metrics[1]]
    colors = ['#bdc3c7', '#27ae60']

    # Gráfico A: Precisión / Convergencia
    plt.figure(figsize=(8, 6))
    bars = plt.bar(strategies, errors, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    plt.title('Precisión de Convergencia (ZDT1 - Convexo)', fontsize=14, fontweight='bold')
    plt.ylabel('Error Promedio (Menor es mejor)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', 
                 ha='center', va='bottom', fontweight='bold')
                 
    plt.savefig(os.path.join(IMG_DIR, "zdt1_comparison_error.png"))
    plt.close()

    # Gráfico B: Cobertura / Spread
    plt.figure(figsize=(8, 6))
    bars = plt.bar(strategies, spreads, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    plt.title('Diversidad en Geometría Convexa (Spread)', fontsize=14, fontweight='bold')
    plt.ylabel('Spread (Mayor es mejor)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', 
                 ha='center', va='bottom', fontweight='bold')

    plt.savefig(os.path.join(IMG_DIR, "zdt1_comparison_spread.png"))
    plt.close()
    print(f"✅ Comparativas generadas en: {IMG_DIR}")

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    # 1. Ejecutar Baseline
    metrics_sbx = run_experiment_zdt1('SBX')
    
    # 2. Ejecutar Propuesta
    metrics_de = run_experiment_zdt1('DE')
    
    # 3. Sintetizar Comparación
    generate_comparative_charts_zdt1(metrics_sbx, metrics_de)
    
    print("\n¡Material para ZDT1 (Convexo) generado exitosamente!")
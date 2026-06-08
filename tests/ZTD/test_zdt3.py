import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend estable para generación de archivos
import matplotlib.pyplot as plt

# --- IMPORTACIÓN DE DEPENDENCIAS ---
from moead.crossovers import SBXCrossover
from moead.mutations import PolynomialMutation
from moead.algorithms import MOEAD
from moead.problems import Problem, ZDT3Problem
from moead.scalarizations import PBI, Tchebycheff
from moead.evolutionary_operator import CrossoverMutation, DifferentialEvolution

# --- CONFIGURACIÓN DEL ENTORNO ---
IMG_DIR = "seminar_results_zdt3"
os.makedirs(IMG_DIR, exist_ok=True)

# ==============================================================================
# BLOQUE 1: MOTORES DE RENDERIZADO (VISUALIZACIÓN INDIVIDUAL)
# ==============================================================================

def plot_pareto_front(pareto_front, title, filename):
    """
    Genera la validación geométrica para ZDT3.
    CRÍTICO: Muestra las discontinuidades (islas). El algoritmo debe poblar las curvas rojas
    y dejar vacíos los espacios entre ellas.
    """
    save_path = os.path.join(IMG_DIR, filename)
    
    # Extraer objetivos
    f1_alg = [sol.objectives[0] for sol in pareto_front]
    f2_alg = [sol.objectives[1] for sol in pareto_front]

    # Generar Frente Real Teórico (ZDT3: Discontinuo)
    # f2 = 1 - sqrt(f1) - f1 * sin(10*pi*f1)
    x1_real = np.linspace(0, 1, 2000) # Alta densidad para dibujar bien las curvas
    f1_real = x1_real
    f2_real = 1 - np.sqrt(f1_real) - f1_real * np.sin(10 * np.pi * f1_real)
    
    # Filtro visual: En teoría, las partes dominadas deberían eliminarse del frente "Real",
    # pero para efectos visuales, la ecuación completa ayuda a ver la "forma" de la función.
    # Sin embargo, para limpieza, solemos dibujar la curva completa suavemente.

    plt.figure(figsize=(10, 7))
    
    # Capa 1: Soluciones del Algoritmo (Puntos Azules)
    plt.scatter(f1_alg, f2_alg, s=20, c='blue', alpha=0.6, label='Solución Obtenida', zorder=2)
    
    # Capa 2: Frente Teórico (Línea Roja)
    plt.plot(f1_real, f2_real, 'r-', linewidth=1.5, alpha=0.5, label='Frente Teórico (Geometría)', zorder=1)
    
    plt.title(f'Validación de Cobertura Discontinua: {title}')
    plt.xlabel('Objetivo 1 (Minimizar)')
    plt.ylabel('Objetivo 2 (Minimizar)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ajuste de ejes: ZDT3 tiene valores negativos en f2 (los valles del seno)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.8, 1.2) # Rango extendido para ver los valles profundos
    
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
    ax2.set_title('Retención de Soluciones (Diversidad)')
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

def calculate_metrics_zdt3(pareto_front):
    """
    KPIs específicos para ZDT3.
    """
    if not pareto_front:
        return 10.0, 0.0

    total_error = 0
    f1_vals = []
    
    for sol in pareto_front:
        f1, f2 = sol.objectives
        f1_vals.append(f1)
        
        # Frente ideal ZDT3: f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1)
        f1_safe = max(f1, 0.0)
        term_sin = f1_safe * np.sin(10 * np.pi * f1_safe)
        f2_ideal = 1 - np.sqrt(f1_safe) - term_sin
        
        # Error vertical absoluto
        total_error += abs(f2 - f2_ideal)
        
    avg_error = total_error / len(pareto_front)
    
    # Spread: En ZDT3 es vital. Si es bajo (< 0.8), significa que faltan islas enteras.
    spread = max(f1_vals) - min(f1_vals) if f1_vals else 0
    
    return avg_error, spread

def run_experiment_zdt3(algorithm_type):
    """
    Orquestador para ZDT3.
    """
    problem = ZDT3Problem(n_vars=30) 
    
    if algorithm_type == 'SBX':
        print("\n--> Iniciando ZDT3 (Discontinuo) con MOEA/D Estándar (SBX)...")
        evo_op = CrossoverMutation(
            crossover=SBXCrossover(eta=15, prob_cross=0.9),
            mutation=PolynomialMutation(eta=20),
            mating_prob=0.9
        )
        label = "Estándar (SBX)"
        file_suffix = "sbx"
        
    elif algorithm_type == 'DE':
        print("\n--> Iniciando ZDT3 (Discontinuo) con MOEA/D Evolución Diferencial (DE)...")
        # DE ayuda a saltar los vacíos entre islas
        evo_op = DifferentialEvolution(F=0.5, CR=0.6)
        label = "Propuesto (DE)"
        file_suffix = "de"
    
    # Solver Config
    solver = MOEAD(
        problem=problem,
        scalarization=Tchebycheff(),
        evolutionary_op=evo_op,
        h_divisions=149, # Más divisiones para asegurar densidad en las islas
        n_neighbors=20,
        n_generations=250, 
        n_r=20 # Reemplazo más alto para permitir que buenas soluciones dominen su vecindario rápido
    )
    
    pareto_front, history = solver.run()
    
    # Visualización
    plot_pareto_front(pareto_front, f"ZDT3 - {label}", f"zdt3_front_{file_suffix}.png")
    plot_convergence_history(history, label, f"zdt3_history_{file_suffix}.png")
    
    # Métricas
    error, spread = calculate_metrics_zdt3(pareto_front)
    print(f"   [Métricas] {label} -> Error: {error:.5f} | Spread: {spread:.5f}")
    
    return error, spread

# ==============================================================================
# BLOQUE 3: SÍNTESIS COMPARATIVA (GENERADOR DE SLIDES)
# ==============================================================================

def generate_comparative_charts_zdt3(sbx_metrics, de_metrics):
    """
    Genera gráficos comparativos enfocados en la Cobertura (Spread).
    """
    print("\n--> Generando gráficos comparativos de alto nivel para ZDT3...")
    
    strategies = ['MOEA/D-SBX\n(Estándar)', 'MOEA/D-DE\n(Propuesto)']
    errors = [sbx_metrics[0], de_metrics[0]]
    spreads = [sbx_metrics[1], de_metrics[1]]
    colors = ['#bdc3c7', '#27ae60']

    # Gráfico A: Precisión de Ajuste
    plt.figure(figsize=(8, 6))
    bars = plt.bar(strategies, errors, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    plt.title('Precisión de Convergencia (ZDT3 - Discontinuo)', fontsize=14, fontweight='bold')
    plt.ylabel('Error Promedio (Menor es mejor)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', 
                 ha='center', va='bottom', fontweight='bold')
                 
    plt.savefig(os.path.join(IMG_DIR, "zdt3_comparison_error.png"))
    plt.close()

    # Gráfico B: Cobertura de Islas (Spread) - EL MÁS IMPORTANTE
    plt.figure(figsize=(8, 6))
    bars = plt.bar(strategies, spreads, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    plt.title('Cobertura de Islas Disconexas (Spread)', fontsize=14, fontweight='bold')
    plt.ylabel('Spread (Mayor es mejor - Máx ~0.85/1.0)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', 
                 ha='center', va='bottom', fontweight='bold')

    plt.savefig(os.path.join(IMG_DIR, "zdt3_comparison_spread.png"))
    plt.close()
    print(f"✅ Comparativas generadas en: {IMG_DIR}")

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    # 1. Ejecutar Baseline
    metrics_sbx = run_experiment_zdt3('SBX')
    
    # 2. Ejecutar Propuesta
    metrics_de = run_experiment_zdt3('DE')
    
    # 3. Sintetizar Comparación
    generate_comparative_charts_zdt3(metrics_sbx, metrics_de)
    
    print("\n¡Material para ZDT3 (Discontinuo) generado exitosamente!")
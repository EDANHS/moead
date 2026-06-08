import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend no interactivo para guardar archivos
import matplotlib.pyplot as plt

# --- IMPORTACIÓN DE DEPENDENCIAS ---
from moead.crossovers import SBXCrossover
from moead.mutations import PolynomialMutation
from moead.algorithms import MOEAD
from moead.problems import Problem, ZDT6Problem
from moead.scalarizations import PBI, Tchebycheff
from moead.evolutionary_operator import CrossoverMutation, DifferentialEvolution

# --- CONFIGURACIÓN DEL ENTORNO ---
IMG_DIR = "seminar_results_zdt6"
os.makedirs(IMG_DIR, exist_ok=True)

# ==============================================================================
# BLOQUE 1: MOTORES DE RENDERIZADO (VISUALIZACIÓN INDIVIDUAL)
# ==============================================================================

def plot_pareto_front(pareto_front, title, filename):
    """
    Genera la validación geométrica de la solución frente al Frente Real de ZDT6.
    Propósito: Demostrar visualmente la adherencia al frente cóncavo.
    """
    save_path = os.path.join(IMG_DIR, filename)
    
    # Extraer objetivos obtenidos
    f1_alg = [sol.objectives[0] for sol in pareto_front]
    f2_alg = [sol.objectives[1] for sol in pareto_front]

    # Generar Frente Real Teórico (ZDT6: f2 = 1 - f1^2)
    # ZDT6 es cóncavo. Generamos puntos densos para una curva suave.
    x1_real = np.linspace(0, 1, 1000)
    # Nota: La densidad de puntos en ZDT6 no es uniforme, pero la curva geométrica es esta:
    f1_real = 1.0 - np.exp(-4.0 * x1_real) * (np.sin(6.0 * np.pi * x1_real)**6)
    f2_real = 1 - f1_real**2
    
    # Ordenar para ploteo limpio
    sort_idx = np.argsort(f1_real)
    f1_real = f1_real[sort_idx]
    f2_real = f2_real[sort_idx]

    plt.figure(figsize=(10, 7))
    # Capa 1: Soluciones del Algoritmo
    plt.scatter(f1_alg, f2_alg, s=15, c='blue', alpha=0.6, label='Solución Obtenida', zorder=2)
    # Capa 2: Frente Teórico
    plt.plot(f1_real, f2_real, 'r-', linewidth=2, label='Frente Real (ZDT6)', zorder=3)
    
    plt.title(f'Validación Geométrica: {title}')
    plt.xlabel('Objetivo 1 (Minimizar)')
    plt.ylabel('Objetivo 2 (Minimizar)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.2) # Margen superior para ver bien la curva
    
    plt.savefig(save_path)
    plt.close()
    print(f"   [Gráfico] Frente de Pareto guardado en: {save_path}")

def plot_convergence_history(history_log, title, filename):
    """
    Visualiza la dinámica de convergencia y la evolución de la diversidad.
    """
    save_path = os.path.join(IMG_DIR, filename)
    
    if not history_log or 'z_star_per_gen' not in history_log:
        print(f"   [Alerta] No se encontró historial válido para {title}")
        return

    z_star_history = np.array(history_log['z_star_per_gen']).T
    archive_history = history_log['archive_size_per_gen']
    generations = np.arange(len(archive_history))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Subplot 1: Convergencia del Punto Ideal (Z*)
    ax1.plot(generations, z_star_history[0], 'b-', linewidth=1.5, label='Z*_1 (Obj 1)')
    ax1.plot(generations, z_star_history[1], 'g-', linewidth=1.5, label='Z*_2 (Obj 2)')
    ax1.set_title(f'Dinámica de Convergencia: {title}')
    ax1.set_ylabel('Valor del Punto Ideal')
    ax1.grid(True, alpha=0.5)
    ax1.legend()
    
    # Subplot 2: Evolución del Archive (Diversidad)
    ax2.plot(generations, archive_history, 'r-', linewidth=1.5, label='Tamaño del Archive')
    ax2.set_title('Retención de Soluciones No Dominadas')
    ax2.set_xlabel('Generaciones')
    ax2.set_ylabel('Cantidad de Soluciones')
    ax2.grid(True, alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   [Gráfico] Historial guardado en: {save_path}")

# ==============================================================================
# BLOQUE 2: LÓGICA DE NEGOCIO (MÉTRICAS Y EJECUCIÓN)
# ==============================================================================

def calculate_metrics(pareto_front):
    """
    Calcula las KPIs críticas para la comparación: Convergencia (Error) y Diversidad (Spread).
    """
    if not pareto_front:
        return 1.0, 0.0

    # 1. Error de Convergencia (Distancia Euclidiana al frente ZDT6)
    total_error = 0
    f1_vals = []
    for sol in pareto_front:
        f1, f2 = sol.objectives
        f1_vals.append(f1)
        
        # Proyección al frente ideal ZDT6: f2 = 1 - f1^2
        # (Nota: Esta es una simplificación de la distancia perpendicular para eficiencia)
        f2_ideal = 1 - f1**2
        total_error += abs(f2 - f2_ideal)
        
    avg_error = total_error / len(pareto_front)
    
    # 2. Spread (Amplitud de cobertura)
    spread = max(f1_vals) - min(f1_vals) if f1_vals else 0
    
    return avg_error, spread

def run_experiment(algorithm_type):
    """
    Orquestador de ejecución para un algoritmo específico.
    Configura, Ejecuta, Grafica y Mide.
    """
    problem = ZDT6Problem(n_vars=10)
    
    if algorithm_type == 'SBX':
        print("\n--> Iniciando Protocolo de Prueba: MOEA/D Estándar (SBX)...")
        # Configuración Estándar
        evo_op = CrossoverMutation(
            crossover=SBXCrossover(eta=15, prob_cross=0.9),
            mutation=PolynomialMutation(eta=20),
            mating_prob=0.9
        )
        generations = 200
        label = "Estándar (SBX)"
        file_suffix = "sbx"
        
    elif algorithm_type == 'DE':
        print("\n--> Iniciando Protocolo de Prueba: MOEA/D Evolución Diferencial (DE)...")
        # Configuración Propuesta (Mejorada)
        evo_op = DifferentialEvolution(F=0.5, CR=0.6)
        generations = 200
        label = "Propuesto (DE)"
        file_suffix = "de"
    
    # Instanciación del Solver
    solver = MOEAD(
        problem=problem,
        scalarization=Tchebycheff(),
        evolutionary_op=evo_op,
        h_divisions=99,
        n_neighbors=20,
        n_generations=generations,
        n_r=2 # Restringido para mayor presión de selección local
    )
    
    # Ejecución
    pareto_front, history = solver.run()
    
    # Generación de Evidencia Visual (Como lo hacías antes)
    plot_pareto_front(pareto_front, f"ZDT6 - {label}", f"zdt6_front_{file_suffix}.png")
    plot_convergence_history(history, label, f"zdt6_history_{file_suffix}.png")
    
    # Cálculo de Métricas (Lo nuevo)
    error, spread = calculate_metrics(pareto_front)
    print(f"   [Métricas] {label} -> Error: {error:.5f} | Spread: {spread:.5f}")
    
    return error, spread

# ==============================================================================
# BLOQUE 3: SÍNTESIS COMPARATIVA (GENERADOR DE SLIDES)
# ==============================================================================

def generate_comparative_charts(sbx_metrics, de_metrics):
    """
    Genera los gráficos de barras comparativos para el seminario.
    """
    print("\n--> Generando gráficos comparativos de alto nivel...")
    
    strategies = ['MOEA/D-SBX\n(Estándar)', 'MOEA/D-DE\n(Propuesto)']
    errors = [sbx_metrics[0], de_metrics[0]]
    spreads = [sbx_metrics[1], de_metrics[1]]
    colors = ['#bdc3c7', '#27ae60'] # Gris (Baseline) vs Verde (Éxito)

    # Gráfico A: Precisión / Convergencia
    plt.figure(figsize=(8, 6))
    bars = plt.bar(strategies, errors, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    plt.title('Análisis de Convergencia: ZDT6 (No Uniforme)', fontsize=14, fontweight='bold')
    plt.ylabel('Error Promedio (Menor es mejor)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Etiquetas de valor
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', 
                 ha='center', va='bottom', fontweight='bold')
                 
    plt.savefig(os.path.join(IMG_DIR, "zdt6_comparison_error.png"))
    plt.close()

    # Gráfico B: Cobertura / Robustez
    plt.figure(figsize=(8, 6))
    bars = plt.bar(strategies, spreads, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    plt.title('Análisis de Diversidad: ZDT6 (Cobertura)', fontsize=14, fontweight='bold')
    plt.ylabel('Spread del Frente (Mayor es mejor)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', 
                 ha='center', va='bottom', fontweight='bold')

    plt.savefig(os.path.join(IMG_DIR, "zdt6_comparison_spread.png"))
    plt.close()
    print(f"✅ Comparativas generadas en: {IMG_DIR}")

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    # 1. Ejecutar Baseline (SBX)
    metrics_sbx = run_experiment('SBX')
    
    # 2. Ejecutar Propuesta (DE)
    metrics_de = run_experiment('DE')
    
    # 3. Generar Comparativa Final
    generate_comparative_charts(metrics_sbx, metrics_de)
    
    print("\n¡Proceso completado con éxito! Todo el material para el seminario está listo.")
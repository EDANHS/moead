import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend no interactivo
import matplotlib.pyplot as plt

# --- IMPORTACIÓN DE DEPENDENCIAS ---
from moead.crossovers import SBXCrossover
from moead.mutations import PolynomialMutation
from moead.algorithms import MOEAD
from moead.problems import Problem, ZDT4Problem
from moead.scalarizations import PBI, Tchebycheff
from moead.evolutionary_operator import CrossoverMutation, DifferentialEvolution

# --- CONFIGURACIÓN DEL ENTORNO ---
IMG_DIR = "seminar_results_zdt4"
# Imprimimos la ruta absoluta para que sepas exactamente dónde busca guardar
print(f"--> Directorio de imágenes configurado en: {os.path.abspath(IMG_DIR)}")
os.makedirs(IMG_DIR, exist_ok=True)

# ==============================================================================
# BLOQUE 1: MOTORES DE RENDERIZADO (VISUALIZACIÓN INDIVIDUAL)
# ==============================================================================

def plot_pareto_front(pareto_front, title, filename):
    """
    Genera la validación geométrica con EJES DINÁMICOS.
    """
    save_path = os.path.join(IMG_DIR, filename)
    
    # 1. Validación de seguridad: Si no hay frente, avisar y salir.
    if not pareto_front:
        print(f"⚠️ ADVERTENCIA: El frente de Pareto para '{filename}' está vacío. No se generará gráfico.")
        return

    # Extraer objetivos
    f1_alg = [sol.objectives[0] for sol in pareto_front]
    f2_alg = [sol.objectives[1] for sol in pareto_front]

    # Generar Frente Real Teórico (ZDT4 es convexo: f2 = 1 - sqrt(f1))
    x1_real = np.linspace(0, 1, 500)
    f1_real = x1_real
    f2_real = 1 - np.sqrt(f1_real)
    
    plt.figure(figsize=(10, 7))
    
    # Capa 1: Soluciones del Algoritmo
    plt.scatter(f1_alg, f2_alg, s=25, c='blue', alpha=0.6, label='Solución Obtenida', zorder=2)
    
    # Capa 2: Frente Teórico
    plt.plot(f1_real, f2_real, 'r--', linewidth=2.5, label='Frente Global Real (ZDT4)', zorder=3)
    
    plt.title(f'Validación de Convergencia Global: {title}')
    plt.xlabel('Objetivo 1 (Minimizar)')
    plt.ylabel('Objetivo 2 (Minimizar)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # --- CORRECCIÓN DE EJES (DINÁMICO) ---
    # ZDT4 es traicionero. Si el algoritmo falla, los puntos pueden estar muy lejos.
    # Calculamos los límites basándonos en los datos reales obtenidos.
    
    max_f1_alg = max(f1_alg) if f1_alg else 1.0
    max_f2_alg = max(f2_alg) if f2_alg else 1.0
    
    # El límite será el máximo entre 1.1 (teórico con margen) y lo que encontró el algoritmo
    limit_x = max(1.1, max_f1_alg * 1.05)
    limit_y = max(1.1, max_f2_alg * 1.05)
    
    plt.xlim(-0.05, limit_x)
    plt.ylim(-0.1, limit_y) 
    
    plt.savefig(save_path)
    plt.close()
    print(f"   ✅ [Gráfico] Guardado correctamente en: {save_path}")

def plot_convergence_history(history_log, title, filename):
    save_path = os.path.join(IMG_DIR, filename)
    
    if not history_log or 'z_star_per_gen' not in history_log:
        return

    z_star_history = np.array(history_log['z_star_per_gen']).T
    archive_history = history_log['archive_size_per_gen']
    generations = np.arange(len(archive_history))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(generations, z_star_history[0], 'b-', label='Z*_1')
    ax1.plot(generations, z_star_history[1], 'g-', label='Z*_2')
    ax1.set_title(f'Dinámica de Convergencia: {title}')
    ax1.set_ylabel('Punto Ideal')
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    
    ax2.plot(generations, archive_history, 'r-', label='Tamaño Archive')
    ax2.set_title('Evolución de la Población No Dominada')
    ax2.set_xlabel('Generaciones')
    ax2.set_ylabel('Nº Soluciones')
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   ✅ [Gráfico] Historial guardado en: {save_path}")

# ==============================================================================
# BLOQUE 2: LÓGICA DE NEGOCIO
# ==============================================================================

def calculate_metrics_zdt4(pareto_front):
    if not pareto_front:
        return 10.0, 0.0

    total_error = 0
    f1_vals = []
    
    for sol in pareto_front:
        f1, f2 = sol.objectives
        f1_vals.append(f1)
        # Frente ideal ZDT4
        f1_safe = max(f1, 0.0)
        f2_ideal = 1 - np.sqrt(f1_safe)
        total_error += abs(f2 - f2_ideal)
        
    avg_error = total_error / len(pareto_front)
    spread = max(f1_vals) - min(f1_vals) if f1_vals else 0
    return avg_error, spread

def run_experiment_zdt4(algorithm_type):
    problem = ZDT4Problem(n_vars=10) 
    
    if algorithm_type == 'SBX':
        print("\n--> Iniciando ZDT4 (Multimodal) con MOEA/D Estándar (SBX)...")
        evo_op = CrossoverMutation(
            crossover=SBXCrossover(eta=15, prob_cross=0.9),
            mutation=PolynomialMutation(eta=20),
            mating_prob=0.9
        )
        label = "Estándar (SBX)"
        file_suffix = "sbx"
        
    elif algorithm_type == 'DE':
        print("\n--> Iniciando ZDT4 (Multimodal) con MOEA/D Evolución Diferencial (DE)...")
        evo_op = DifferentialEvolution(F=0.5, CR=0.6)
        label = "Propuesto (DE)"
        file_suffix = "de"
    
    solver = MOEAD(
        problem=problem,
        scalarization=Tchebycheff(),
        evolutionary_op=evo_op,
        h_divisions=99,
        n_neighbors=20, 
        n_generations=400,
        n_r=2 
    )
    
    # Ejecutamos el solver
    pareto_front, history = solver.run()
    
    # 3. Validación Extra: ¿Encontró algo?
    if not pareto_front:
        print(f"⚠️ ERROR CRÍTICO: El algoritmo {label} devolvió un frente vacío.")
    else:
        print(f"   Info: Se encontraron {len(pareto_front)} soluciones.")

    # Visualización
    plot_pareto_front(pareto_front, f"ZDT4 - {label}", f"zdt4_front_{file_suffix}.png")
    plot_convergence_history(history, label, f"zdt4_history_{file_suffix}.png")
    
    # Métricas
    error, spread = calculate_metrics_zdt4(pareto_front)
    print(f"   [Métricas] {label} -> Error Global: {error:.5f} | Spread: {spread:.5f}")
    
    return error, spread

# ==============================================================================
# BLOQUE 3: SÍNTESIS COMPARATIVA
# ==============================================================================

def generate_comparative_charts_zdt4(sbx_metrics, de_metrics):
    print("\n--> Generando gráficos comparativos de alto nivel para ZDT4...")
    
    strategies = ['MOEA/D-SBX\n(Estándar)', 'MOEA/D-DE\n(Propuesto)']
    errors = [sbx_metrics[0], de_metrics[0]]
    spreads = [sbx_metrics[1], de_metrics[1]]
    colors = ['#bdc3c7', '#27ae60']

    # Gráfico A: Error
    plt.figure(figsize=(8, 6))
    bars = plt.bar(strategies, errors, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    plt.title('Capacidad de Escape de Óptimos Locales (ZDT4)', fontsize=14, fontweight='bold')
    plt.ylabel('Error Promedio', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', 
                 ha='center', va='bottom', fontweight='bold')
    plt.savefig(os.path.join(IMG_DIR, "zdt4_comparison_error.png"))
    plt.close()

    # Gráfico B: Spread
    plt.figure(figsize=(8, 6))
    bars = plt.bar(strategies, spreads, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    plt.title('Diversidad en Espacio de Objetivos (ZDT4)', fontsize=14, fontweight='bold')
    plt.ylabel('Spread', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', 
                 ha='center', va='bottom', fontweight='bold')
    plt.savefig(os.path.join(IMG_DIR, "zdt4_comparison_spread.png"))
    plt.close()
    print(f"✅ Comparativas generadas en: {IMG_DIR}")

if __name__ == "__main__":
    # Asegúrate de que esto se ejecuta
    metrics_sbx = run_experiment_zdt4('SBX')
    metrics_de = run_experiment_zdt4('DE')
    generate_comparative_charts_zdt4(metrics_sbx, metrics_de)
    print("\n¡Material para ZDT4 generado exitosamente!")
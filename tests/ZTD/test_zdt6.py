import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from crossovers import SBXCrossover
from mutations import PolynomialMutation
from algorithms import MOEAD
from problems import ZDT6Problem
from scalarizations import Tchebycheff, PBI

IMG_DIR = ".images"
os.makedirs(IMG_DIR, exist_ok=True)

def test_zdt6_run():
    """Ejecuta y grafica el test para ZDT6."""
    print("Iniciando test ZDT6...")
    
    # 1. Configurar el problema
    problem = ZDT6Problem(n_vars=10) # <-- CAMBIO (ZDT6 usa n=10)
    crossover = SBXCrossover(eta=15, prob_cross=0.9)
    mutation = PolynomialMutation(eta=20)
    scalarization = Tchebycheff()
    #scalarization = PBI(theta=5.0)
    
    # 2. Inyectar dependencias
    moead_solver = MOEAD(
        problem=problem,
        crossover=crossover,
        mutation=mutation,
        scalarization=scalarization,
        h_divisions=99,
        n_neighbors=20,
        n_generations=200
    )
    
    # 3. Ejecutar
    final_population = moead_solver.run()
    
    # 4. Procesar y graficar
    f1_alg = [sol.objectives[0] for sol in final_population]
    f2_alg = [sol.objectives[1] for sol in final_population]

    # El frente real de ZDT6 es más complejo de graficar
    x1_real = np.linspace(0, 1, 1000)
    f1_real = 1.0 - np.exp(-4.0 * x1_real) * (np.sin(6.0 * np.pi * x1_real)**6)
    f2_real = 1 - f1_real**2
    
    # Ordenar para un ploteo correcto
    sort_indices = np.argsort(f1_real)
    f1_real_sorted = f1_real[sort_indices]
    f2_real_sorted = f2_real[sort_indices]

    plt.figure(figsize=(10, 7))
    plt.scatter(f1_alg, f2_alg, s=15, c='blue', label='Resultado MOEA/D', zorder=2)
    plt.plot(f1_real_sorted, f2_real_sorted, 'r', linewidth=2, label='Frente Real ZDT6', zorder=1)
    plt.title('Verificación del Frente de Pareto (ZDT6)')
    plt.xlabel('Objetivo 1 (f1)')
    plt.ylabel('Objetivo 2 (f2)')
    plt.legend()
    plt.grid(True)

    plt.ylim(bottom=-0.1, top=1.5)
    
    # 5. Guardar la imagen
    save_path = os.path.join(IMG_DIR, "zdt6_front.png") # <-- CAMBIO
    plt.savefig(save_path)
    plt.close()
    
    print(f"Test ZDT6 completado. Imagen guardada en: {save_path}")

if __name__ == "__main__":
    test_zdt6_run()
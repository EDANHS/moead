import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from crossovers import SBXCrossover
from mutations import PolynomialMutation
from algorithms import MOEAD
from problems import ZDT3Problem
from scalarizations import PBI, Tchebycheff

IMG_DIR = ".images"
os.makedirs(IMG_DIR, exist_ok=True)

def test_zdt3_run():
    """Ejecuta y grafica el test para ZDT3."""
    print("Iniciando test ZDT3...")
    
    # 1. Configurar el problema
    problem = ZDT3Problem(n_vars=30) # <-- CAMBIO
    crossover = SBXCrossover(eta=15, prob_cross=0.9)
    mutation = PolynomialMutation(eta=20)
    scalarization = Tchebycheff()
    #scalarization = PBI(theta=5.0)
    
    # 2. Inyectar dependencias (más generaciones, es más difícil)
    moead_solver = MOEAD(
        problem=problem,
        crossover=crossover,
        mutation=mutation,
        scalarization=scalarization,
        h_divisions=99,
        n_neighbors=20,
        n_generations=250 # <-- Más generaciones
    )
    
    # 3. Ejecutar
    final_population = moead_solver.run()
    
    # 4. Procesar y graficar
    f1_alg = [sol.objectives[0] for sol in final_population]
    f2_alg = [sol.objectives[1] for sol in final_population]

    f1_real = np.linspace(0, 1, 1000) # Más puntos para la curva discontinua
    f2_real = 1 - np.sqrt(f1_real) - f1_real * np.sin(10 * np.pi * f1_real) # <-- CAMBIO

    plt.figure(figsize=(10, 7))
    plt.scatter(f1_alg, f2_alg, s=15, c='blue', label='Resultado MOEA/D', zorder=2)
    plt.plot(f1_real, f2_real, 'r', linewidth=2, label='Frente Real ZDT3', zorder=1) # Línea sólida
    plt.title('Verificación del Frente de Pareto (ZDT3)')
    plt.xlabel('Objetivo 1 (f1)')
    plt.ylabel('Objetivo 2 (f2)')
    plt.legend()
    plt.grid(True)

    plt.ylim(bottom=-1.0, top=1.5)
    
    # 5. Guardar la imagen
    save_path = os.path.join(IMG_DIR, "zdt3_front.png") # <-- CAMBIO
    plt.savefig(save_path)
    plt.close()
    
    print(f"Test ZDT3 completado. Imagen guardada en: {save_path}")

if __name__ == "__main__":
    test_zdt3_run()
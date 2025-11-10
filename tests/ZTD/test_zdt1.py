import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from crossovers import SBXCrossover
from mutations import PolynomialMutation
from algorithms import MOEAD
from problems import ZDT1Problem
from scalarizations import PBI, Tchebycheff

# Directorio para guardar las imágenes
IMG_DIR = ".images"
os.makedirs(IMG_DIR, exist_ok=True)

def test_zdt1_run():
    """Ejecuta y grafica el test para ZDT1."""
    print("Iniciando test ZDT1...")
    
    # 1. Configurar el problema y los operadores
    problem = ZDT1Problem(n_vars=30)
    crossover = SBXCrossover(eta=15, prob_cross=0.9)
    mutation = PolynomialMutation(eta=20)
    scalarization = Tchebycheff()
    #scalarization = PBI(theta=5.0)
    
    # 2. Inyectar dependencias en el solver
    moead_solver = MOEAD(
        problem=problem,
        crossover=crossover,
        mutation=mutation,
        scalarization=scalarization,
        h_divisions=99,    # N=100
        n_neighbors=20,
        n_generations=150
    )
    
    # 3. Ejecutar
    final_population = moead_solver.run()
    
    # 4. Procesar y graficar resultados
    f1_alg = [sol.objectives[0] for sol in final_population]
    f2_alg = [sol.objectives[1] for sol in final_population]

    # Generar el frente de Pareto real
    f1_real = np.linspace(0, 1, 100)
    f2_real = 1 - np.sqrt(f1_real)

    plt.figure(figsize=(10, 7))
    plt.scatter(f1_alg, f2_alg, s=15, c='blue', label='Resultado MOEA/D')
    plt.plot(f1_real, f2_real, 'r--', linewidth=2, label='Frente Real ZDT1')
    plt.title('Verificación del Frente de Pareto (ZDT1)')
    plt.xlabel('Objetivo 1 (f1)')
    plt.ylabel('Objetivo 2 (f2)')
    plt.legend()
    plt.grid(True)
    
    plt.ylim(bottom=-0.1, top=1.5)

    # 5. Guardar la imagen
    save_path = os.path.join(IMG_DIR, "zdt1_front.png")
    plt.savefig(save_path)
    plt.close() # Cerrar la figura para liberar memoria
    
    print(f"Test ZDT1 completado. Imagen guardada en: {save_path}")

if __name__ == "__main__":
    test_zdt1_run()
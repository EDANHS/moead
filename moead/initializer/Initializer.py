from abc import ABC, abstractmethod
from moead.problems import Problem
from moead.utils import History

class Initializer(ABC):
    """
    Contrato abstracto para las estrategias de inicialización de la Generación Cero.
    Garantiza el desacoplamiento entre el orquestador evolutivo y la lógica de prospección.
    """
    @abstractmethod
    def execute(self, problem: Problem, n_pop: int, history: History) -> list:
        """
        Orquesta la creación y evaluación inicial de las soluciones.
        
        :param problem: Instancia del problema de optimización subyacente.
        :param n_pop: Volumen de individuos requeridos para la población.
        :param history: Objeto History con la telemetría de ejecuciones previas.
        :return: Lista de objetos Solution instanciados y evaluados (o pre-evaluados).
        """
        pass
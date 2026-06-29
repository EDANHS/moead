# Nuevo Archivo: DLProblemZCP.py
import numpy as np
import json
from .DLProblemRefactor import DLProblemRefactor
from moead.utils import SurrogatePredictor

class DLProblemZCP(DLProblemRefactor):
    """
    Especialización Zero-Cost del Problema de Optimización.
    Implementa un motor de evaluación híbrido de baja latencia:
    - F1 (Pérdida): Inferencia subrogada vía Random Forest (ZCPSurrogate).
    - F2 (Complejidad): Cálculo algebraico exacto O(1) (Conteo paramétrico).
    """
    def __init__(self, surrogate_model: SurrogatePredictor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surrogate = surrogate_model

    def _calculate_params_analytical(self, config: dict) -> int:
        """
        Cálculo algebraico exacto de los parámetros de la topología U-Net.
        Reemplaza la llamada a Keras/TensorFlow, operando en O(1).
        """
        depth = config['depth']
        filters = config['initial_filters']
        kernel = config['kernel_size'][0] if isinstance(config['kernel_size'], list) else config['kernel_size']
        use_bias = config['use_bias']
        use_bn = (config['norm_type'] == 'Batch')
        
        total_params = 0
        in_channels = self.input_shape[-1]
        
        # 1. Encoder (Bajada)
        current_filters = filters
        for i in range(depth):
            params_conv1 = (kernel * kernel * in_channels * current_filters) + (current_filters if use_bias else 0)
            if use_bn: params_conv1 += (4 * current_filters)
            
            params_conv2 = (kernel * kernel * current_filters * current_filters) + (current_filters if use_bias else 0)
            if use_bn: params_conv2 += (4 * current_filters)
            
            total_params += (params_conv1 + params_conv2)
            in_channels = current_filters
            current_filters *= 2
            
        # 2. Bottleneck (Cuello de botella)
        params_bot1 = (kernel * kernel * in_channels * current_filters) + (current_filters if use_bias else 0)
        if use_bn: params_bot1 += (4 * current_filters)
        
        params_bot2 = (kernel * kernel * current_filters * current_filters) + (current_filters if use_bias else 0)
        if use_bn: params_bot2 += (4 * current_filters)
        
        total_params += (params_bot1 + params_bot2)
        in_channels = current_filters
        
        # 3. Decoder (Subida)
        for i in range(depth):
            current_filters //= 2
            
            if config['upsample_type'] == 'TransposeConv':
                params_up = (2 * 2 * in_channels * current_filters) + (current_filters if use_bias else 0)
                total_params += params_up
                
            in_channels = current_filters * 2 
            
            params_dec1 = (kernel * kernel * in_channels * current_filters) + (current_filters if use_bias else 0)
            if use_bn: params_dec1 += (4 * current_filters)
            
            params_dec2 = (kernel * kernel * current_filters * current_filters) + (current_filters if use_bias else 0)
            if use_bn: params_dec2 += (4 * current_filters)
            
            total_params += (params_dec1 + params_dec2)
            in_channels = current_filters
            
        # 4. Capa de Salida
        out_classes = 1
        params_out = (1 * 1 * in_channels * out_classes) + (out_classes if use_bias else 0)
        total_params += params_out
        
        return total_params

    def evaluate(self, solution):
        """
        Intercepta la evaluación estándar evitando el uso de GPUs / workers pesados.
        """
        try:
            if hasattr(self, 'debugger') and self.debugger:
                self.debugger.start_step('evaluate_solution_zcp', {
                    'variables': solution.variables.tolist() if isinstance(solution.variables, np.ndarray) else solution.variables,
                })

            if not self._is_within_bounds(solution.variables):
                solution.objectives = np.full(self.n_objectives, np.inf)
                solution.constraints = np.full(self.n_constraints, np.inf)
                solution.invalid_genotype = True
                return

            config = self.decode_solution(solution.variables)
            
            # OBJETIVO 1: Inferencia Subrogada (Predicción de Dice Loss)
            predicted_loss = self.surrogate.predict_loss(config)
            
            # OBJETIVO 2: Cálculo Analítico (Volumen de Parámetros Normalizado)
            raw_params = self._calculate_params_analytical(config)
            obj_params_norm = float((raw_params - self.z_min_params) / (self.z_max_params - self.z_min_params))
            obj_params_norm = float(np.clip(obj_params_norm, 0.0, 1.0))
            
            # Asignación atómica de objetivos, evitando compilación de grafos
            solution.objectives = np.array([predicted_loss, obj_params_norm])
            solution.constraints = np.zeros(self.n_constraints)
            solution.set_metadata(config=config, training_time=0.001, epoch=0)

            if hasattr(self, 'debugger') and self.debugger:
                self.debugger.pass_step('evaluate_solution_zcp', 'Evaluación Híbrida ZCP Exitosa')

        except Exception as e:
            if self.verbose >= 1: print(f"    [ZCP ERROR] Fallo en la evaluación híbrida: {e}")
            solution.objectives = np.full(self.n_objectives, np.inf)
            solution.constraints = np.full(self.n_constraints, np.inf)
            solution.invalid_genotype = True
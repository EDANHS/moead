import numpy as np
import json
import time
from .DLProblemRefactor import DLProblemRefactor
from moead.utils import SurrogatePredictor

class DLProblemZCP(DLProblemRefactor):
    """
    Especialización Zero-Cost del Problema de Optimización.
    Mantiene compatibilidad de logs con el ecosistema de visualización 
    y reportabilidad de métricas (Dice, Parámetros, Latencia real).
    """
    def __init__(self, surrogate_model: SurrogatePredictor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surrogate = surrogate_model

    def _calculate_params_analytical(self, config: dict) -> int:
        """Cálculo algebraico exacto de los parámetros de la topología U-Net."""
        depth = config['depth']
        filters = config['initial_filters']
        
        # SANEAMIENTO DE SEGURIDAD (FIX ROBUSTO):
        k_val = config.get('kernel_size', 3)
        kernel = int(k_val[0]) if isinstance(k_val, (list, tuple)) else int(k_val)
        
        use_bias = config['use_bias']
        use_bn = (config['norm_type'] == 'Batch')
        
        total_params = 0
        in_channels = self.input_shape[-1]
        
        # Encoder
        current_filters = filters
        for i in range(depth):
            # Usamos el escalar 'kernel' ya sanitizado
            params_conv1 = (kernel * kernel * in_channels * current_filters) + (current_filters if use_bias else 0)
            if use_bn: params_conv1 += (4 * current_filters)
            params_conv2 = (kernel * kernel * current_filters * current_filters) + (current_filters if use_bias else 0)
            if use_bn: params_conv2 += (4 * current_filters)
            total_params += (params_conv1 + params_conv2)
            in_channels = current_filters
            current_filters *= 2
            
        # Bottleneck
        params_bot1 = (kernel * kernel * in_channels * current_filters) + (current_filters if use_bias else 0)
        if use_bn: params_bot1 += (4 * current_filters)
        params_bot2 = (kernel * kernel * current_filters * current_filters) + (current_filters if use_bias else 0)
        if use_bn: params_bot2 += (4 * current_filters)
        total_params += (params_bot1 + params_bot2)
        in_channels = current_filters
        
        # Decoder
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
            
        # Salida
        total_params += (1 * 1 * in_channels * 1) + (1 if use_bias else 0)
        return int(total_params)

    def evaluate(self, solution):
        """
        Evaluación híbrida que mide el tiempo real de ejecución para
        compatibilidad con el script de graficación y análisis de latencia.
        """
        try:
            start_eval_time = time.perf_counter()
            config = self.decode_solution(solution.variables)


            if self.verbose >= 1: print(f"\n--> [CACHE MISS] Evaluando arquitectura: {config}")

            if hasattr(solution, 'zcp_metrics'):
                config.update(solution.zcp_metrics)

            # Cálculo de métricas
            predicted_loss = float(self.surrogate.predict_loss(config))
            raw_params = self._calculate_params_analytical(config)
            
            # Normalización
            obj_dice_loss = float(predicted_loss)
            obj_params_norm = float(np.clip((raw_params - self.z_min_params) / (self.z_max_params - self.z_min_params), 0.0, 1.0))
            
            elapsed_time = time.perf_counter() - start_eval_time
            
            # Print formateado para el script de graficación con tiempo real
            if self.verbose >= 1:
                print(f"    Resultados -> Dice Loss: {obj_dice_loss:.4f} | Params Norm: {obj_params_norm:.4f} | Tiempo: {elapsed_time:.6f}s")
                print(f"    [OK] ZCP-Synflow: {solution.zcp_metrics['zcp_synflow']:.2e} | ZCP-SNIP: {solution.zcp_metrics['zcp_snip']:.2e} | ZCP-Jacobian: {solution.zcp_metrics['zcp_jacobian']:.2e}")
            solution.objectives = np.array([obj_dice_loss, obj_params_norm])
            solution.constraints = np.zeros(self.n_constraints)
            # metadata requerida por el visualizador
            solution.set_metadata(config=config, training_time=elapsed_time, epoch=0)

        except Exception as e:
            if self.verbose >= 1: print(f"    [ZCP ERROR] Fallo en evaluación: {e}")
            solution.objectives = np.array([1.0, 1.0])
            solution.constraints = np.full(self.n_constraints, np.inf)
            solution.invalid_genotype = True
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class SurrogatePredictor:
    """
    Oráculo de Predicción (Zero-Cost Surrogate).
    Transforma el historial empírico de evaluaciones previas en un modelo
    de regresión no lineal capaz de estimar la pérdida (Dice Loss) instantáneamente.
    """
    def __init__(self, history_file_path: str):
        self.history_file_path = history_file_path
        self.is_trained = False
        
        # Instanciamos el regresor. 
        # Random Forest es ideal porque maneja muy bien relaciones no lineales 
        # y no requiere escalar/normalizar los datos de entrada rigurosamente.
        self.model = RandomForestRegressor(n_estimators=100, 
                                           max_depth=None, 
                                           random_state=42, 
                                           n_jobs=-1) # Uso de todos los núcleos del CPU
        
        # Mapeos categóricos basados en tu DLProblemRefactor
        self.act_opts = ['ReLU', 'ELU', 'LeakyReLU', 'GELU', 'Swish']
        self.norm_opts = ['Batch', 'Layer', 'Instance', 'None']
        self.pool_opts = ['Max', 'Average']
        self.upsample_opts = ['TransposeConv', 'BilinearUpsample']

    def _vectorize_config(self, config: dict) -> np.ndarray:
        """
        Convierte el diccionario de configuración en un vector numérico 1D.
        Aplica Label Encoding para las variables categóricas.
        """
        # Extraemos el valor escalar del kernel (asumiendo que es cuadrado, ej. [5,5] -> 5)
        kernel_val = config['kernel_size'][0] if isinstance(config['kernel_size'], list) else config['kernel_size']
        
        features = [
            float(config['depth']),
            float(config['initial_filters']),
            float(kernel_val),
            float(self.act_opts.index(config['activation_name'])), # Convierte a índice entero
            float(self.norm_opts.index(config['norm_type'])),
            float(config['dropout_rate']),
            float(1.0 if config['use_bias'] else 0.0),             # Boolean a Float
            float(self.pool_opts.index(config['pooling_type'])),
            float(self.upsample_opts.index(config['upsample_type']))
        ]
        return np.array(features)

    def train_surrogate(self, verbose: int = 1):
        """
        Lee el historial JSON, extrae la métrica objetivo (Dice Loss) y 
        entrena el ensamble predictivo.
        """
        if not os.path.exists(self.history_file_path):
            raise FileNotFoundError(f"[ERROR] No se encontró el historial base en: {self.history_file_path}. El ZCP Subrogado requiere datos previos para funcionar.")
            
        if verbose >= 1:
            print(f"\n--> [SURROGATE] Entrenando Oráculo Predictivo desde: {self.history_file_path}")
            
        with open(self.history_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            
        X = []
        y = []
        
        for config_str, metrics in cache_data.items():
            try:
                # 1. Recuperar el diccionario a partir del string JSON
                config_dict = json.loads(config_str)
                
                # 2. Vectorizar características (Genotipo)
                vectorized_features = self._vectorize_config(config_dict)
                
                # 3. Extraer el Dice Loss real (Fenotipo)
                # Asumimos que objectives[0] es la pérdida a minimizar y validamos que no sea un error (inf)
                dice_loss = metrics['objectives'][0]
                if np.isinf(dice_loss) or np.isnan(dice_loss):
                    continue # Descartamos redes que fallaron por OOM o gradientes explosivos
                    
                X.append(vectorized_features)
                y.append(dice_loss)
                
            except Exception as e:
                if verbose >= 2:
                    print(f"  [WARN] Imposible parsear entrada histórica: {e}")
                continue
                
        if len(X) < 10:
            raise ValueError("[ERROR] Insuficientes datos válidos en el historial para entrenar el modelo subrogado (Mínimo recomendado: 10).")

        X = np.array(X)
        y = np.array(y)
        
        # 4. Ajuste del Modelo Predictivo
        self.model.fit(X, y)
        self.is_trained = True
        
        if verbose >= 1:
            # Una evaluación de sanidad interna para saber qué tan bien aprendió la topología de los datos
            train_preds = self.model.predict(X)
            mse = mean_squared_error(y, train_preds)
            print(f"--> [SURROGATE] Entrenado exitosamente con {len(X)} arquitecturas. MSE Interno: {mse:.6f}\n")

    def predict_loss(self, config: dict) -> float:
        """
        Inferencia ultrarrápida. Dado un genotipo, predice su Dice Loss en milisegundos.
        """
        if not self.is_trained:
            raise RuntimeError("[ERROR] Intento de predicción sin haber entrenado el modelo subrogado.")
            
        # Vectorizamos la nueva configuración de la misma forma que en el entrenamiento
        features = self._vectorize_config(config)
        
        # model.predict requiere un array 2D, por lo que encapsulamos features en una lista
        predicted_loss = self.model.predict([features])[0]
        
        # Agregamos una capa de seguridad para mantener la predicción dentro de los límites del Dice Loss (0 a 1)
        return float(np.clip(predicted_loss, 0.0, 1.0))
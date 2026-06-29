import json
import joblib
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class SurrogatePredictor:
    """
    Oráculo de Predicción (Zero-Cost Surrogate).
    Mapea el genotipo y los componentes analíticos de una U-Net para
    predecir su Dice Loss de forma instantánea sin compilar grafos.
    """
    def __init__(self, history_file_path: str = None, model_path: str = None):
        """
        :param history_file_path: Ruta del JSON maestro unificado.
        :param model_path: Ruta opcional para guardar/cargar el modelo serializado (.pkl).
        """
        self.history_file_path = history_file_path
        self.model_path = model_path
        self.is_trained = False
        
        # Configuración del regresor optimizado para CPU multi-núcleo
        self.model = RandomForestRegressor(n_estimators=200, 
                                           max_depth=None, 
                                           random_state=42, 
                                           n_jobs=-1)
        
        # Mapeos categóricos estructurales
        self.act_opts = ['ReLU', 'ELU', 'LeakyReLU', 'GELU', 'Swish']
        self.norm_opts = ['Batch', 'Layer', 'Instance', 'None']
        self.pool_opts = ['Max', 'Average']
        self.upsample_opts = ['TransposeConv', 'BilinearUpsample']

        # Intentar carga automática si se provee un binario existente
        if self.model_path and os.path.exists(self.model_path):
            self._load_model_binary()

    def _normalize_kernel(self, kernel_input) -> int:
        """
        Saneamiento geométrico: Convierte tuplas de kernel (e.g., (3,3)) a su 
        equivalente escalar (e.g., 3). Garantiza compatibilidad matemática
        con los cálculos de flops y parámetros.
        """
        if isinstance(kernel_input, (list, tuple)):
            # Extrae el primer elemento si es un par, o el escalar si es entero
            return int(kernel_input[0])
        return int(kernel_input)

    def _vectorize_config(self, config: dict) -> np.ndarray:
        """
        Estrategia de Transformación: Mapea diccionarios genotípicos a tensores de 
        características estables. La normalización del kernel es crítica para 
        evitar errores de tipo en DLProblemZCP.
        """
        k_val = self._normalize_kernel(config.get('kernel_size', 3))
        
        # Bloque de Características Estructurales (Fase Metodológica)
        features = [
            float(config['depth']),
            float(config['initial_filters']),
            float(k_val),
            float(self.act_opts.index(config.get('activation_name', 'ReLU'))),
            float(self.norm_opts.index(config.get('norm_type', 'Batch'))),
            float(config.get('dropout_rate', 0.0)),
            float(1.0 if config.get('use_bias', False) else 0.0),
            float(self.pool_opts.index(config.get('pooling_type', 'Max'))),
            float(self.upsample_opts.index(config.get('upsample_type', 'TransposeConv')))
        ]
        
        # Inyección de Métricas ZCP (Zero-Cost Proxies) para robustez multidimensional
        features.extend([
            float(config.get('zcp_synflow', 0.0)),
            float(config.get('zcp_snip', 0.0)),
            float(config.get('zcp_jacobian', 0.0))
        ])
            
        return np.array(features)

    def train_surrogate(self, verbose: int = 1):
        """
        Metodología de entrenamiento del subrogado con purga de anomalías.
        Utiliza el caché persistente del ecosistema para inferir Dice Loss.
        """
        if not self.history_file_path or not os.path.exists(self.history_file_path):
            raise FileNotFoundError("[ERROR] No se localizó el archivo de caché histórico.")

        with open(self.history_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        X, y = [], []
        
        for config_str, metrics in cache_data.items():
            try:
                config_dict = json.loads(config_str)
                # Integrar métricas ZCP al diccionario de entrenamiento
                config_dict.update({k: v for k, v in metrics.items() if 'zcp' in k})
                
                row_features = self._vectorize_config(config_dict)
                
                # Validación de estabilidad numérica
                if np.any(np.isnan(row_features)) or np.any(np.isinf(row_features)):
                    continue
                
                dice_loss = metrics['objectives'][0]
                if 0.0 <= dice_loss <= 1.0:
                    X.append(row_features)
                    y.append(dice_loss)
            except Exception:
                continue

        X, y = np.array(X), np.array(y)
        self.model.fit(X, y)
        self.is_trained = True
        
        if self.model_path:
            self._save_model_binary()
        
        if verbose >= 1:
            print(f"--> [SURROGATE] Entrenamiento consolidado con {len(X)} arquitecturas validadas.")


    def predict_loss(self, config: dict) -> float:
        """Inferencia de Dice Loss con clip de seguridad [0, 1]."""
        if not self.is_trained:
            return 0.5 # Valor neutro si el modelo aún no está hidratado
            
        features = self._vectorize_config(config)
        
        features_2d = features.reshape(1, -1)
        print(f"--> [SURROGATE] Prediciendo Dice Loss para configuración: {config}")
        prediction = self.model.predict(features_2d)
        
        return float(np.clip(prediction[0], 0.0, 1.0))

    def _load_model_binary(self):
        """Carga el modelo usando Joblib con validación de tipo de archivo."""
        try:
            # Verificación preventiva: no intentar cargar si es un archivo de texto/json
            with open(self.model_path, 'rb') as f:
                header = f.read(5)
                # Joblib suele empezar con ciertos bytes binarios, si empieza con '{' es JSON corrupto
                if header.startswith(b'{'):
                    raise ValueError("El archivo detectado es JSON, no un binario de modelo.")
            
            state = joblib.load(self.model_path)
            self.model = state['model']
            self.is_trained = state['is_trained']
            print(f"--> [INFO] Modelo cargado correctamente desde {self.model_path}")
            
        except (Exception) as e:
            print(f"--> [ERROR] Modelo corrupto o incompatible en {self.model_path}. Detalle: {e}")
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            self.is_trained = False
            self.model = RandomForestRegressor(n_estimators=250, max_depth=None, random_state=42, n_jobs=-1)

    def _save_model_binary(self):
        """Guarda de forma segura usando Joblib."""
        if not self.model_path:
            return
        temp_path = self.model_path + ".tmp"
        joblib.dump({'model': self.model, 'is_trained': self.is_trained}, temp_path)
        os.replace(temp_path, self.model_path)
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

    def _vectorize_config(self, config: dict) -> np.ndarray:
        """
        Transforma el diccionario genotípico en un vector numérico plano.
        Soporta de forma adaptativa la inclusión o ausencia de métricas ZCP.
        """
        # --- FIX ROBUSTO DE EXTRACCIÓN DE KERNEL ---
        raw_kernel = config.get('kernel_size', 3)
        if isinstance(raw_kernel, (list, tuple)):
            kernel_val = float(raw_kernel[0]) # Tomamos la primera dimensión si es iterable
        else:
            kernel_val = float(raw_kernel)
        # -------------------------------------------
        
        # 1. Características base de la topología (Grados de libertad)
        features = [
            float(config['depth']),
            float(config['initial_filters']),
            kernel_val,
            float(self.act_opts.index(config.get('activation_name', 'ReLU'))),
            float(self.norm_opts.index(config.get('norm_type', 'Batch'))),
            float(config.get('dropout_rate', 0.0)),
            float(1.0 if config.get('use_bias', False) else 0.0),
            float(self.pool_opts.index(config.get('pooling_type', 'Max'))),
            float(self.upsample_opts.index(config.get('upsample_type', 'TransposeConv')))
        ]
        
        # 2. Inclusión dinámica de métricas ZCP si se encuentran en el diccionario
        # Esto permite que la misma clase sirva para el MVP y para la versión avanzada con Jacobiano
        if 'zcp_synflow' in config: features.append(float(config['zcp_synflow']))
        if 'zcp_snip' in config: features.append(float(config['zcp_snip']))
        if 'zcp_jacobian' in config: features.append(float(config['zcp_jacobian']))
            
        return np.array(features)

    def train_surrogate(self, verbose: int = 1):
        """
        Extrae los datos del caché, entrena el bosque correlacionando la estructura
        con el Dice Loss real y guarda el estado si se especificó una ruta binaria.
        """
        if self.is_trained:
            if verbose >= 1: print("[*] El modelo subrogado ya se encuentra operativo en memoria.")
            return

        if not self.history_file_path or not os.path.exists(self.history_file_path):
            raise FileNotFoundError(f"[ERROR] Archivo de caché histórico no parametrizado o inexistente.")

        if verbose >= 1:
            print(f"\n--> [SURROGATE] Entrenando Oráculo Predictivo desde: {self.history_file_path}")

        with open(self.history_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        X, y = [], []
        
        for config_str, metrics in cache_data.items():
            try:
                config_dict = json.loads(config_str)
                
                # Sincronizamos las métricas calculadas del nivel fenotípico al diccionario estructural
                if 'zcp_synflow' in metrics:
                    config_dict['zcp_synflow'] = metrics['zcp_synflow']
                    config_dict['zcp_snip'] = metrics['zcp_snip']
                    config_dict['zcp_jacobian'] = metrics['zcp_jacobian']
                
                dice_loss = metrics['objectives'][0]
                if np.isinf(dice_loss) or np.isnan(dice_loss):
                    continue
                
                X.append(self._vectorize_config(config_dict))
                y.append(dice_loss)
            except Exception:
                continue

        X, y = np.array(X), np.array(y)
        
        # Ajuste adaptativo del Random Forest
        self.model.fit(X, y)
        self.is_trained = True

        if verbose >= 1:
            train_preds = self.model.predict(X)
            mse = mean_squared_error(y, train_preds)
            print(f"--> [SURROGATE] Sintonización completada con {len(X)} muestras. MSE de ajuste: {mse:.6f}")

        # Serialización defensiva para evitar re-entrenamientos futuros
        if self.model_path:
            self._save_model_binary()

    def predict_loss(self, config: dict) -> float:
        """
        Ejecuta inferencia subrogada en O(1) estimando el Dice Loss.
        """
        if not self.is_trained:
            raise RuntimeError("[ERROR] Instancia del predictor no entrenada ni rehidratada.")
            
        features = self._vectorize_config(config)
        predicted_loss = self.model.predict([features])[0]
        return float(np.clip(predicted_loss, 0.0, 1.0))

    def _save_model_binary(self):
        """Guarda el objeto estructurado de scikit-learn en un archivo binario."""
        
        # Asegurarse de que el directorio exista
        directorio = os.path.dirname(self.model_path)
        if directorio:
            os.makedirs(directorio, exist_ok=True)
            
        estado = {
            'model': self.model,
            'is_trained': self.is_trained,
            'feature_len': len(self.model.feature_importances_)
        }
        
        # Usamos joblib con compress=3 para mantener el archivo ligero
        joblib.dump(estado, self.model_path, compress=3)
        print(f"[+] Estado del binario persistido de forma atómica en: {self.model_path}")

    def _load_model_binary(self):
        """Carga e inyecta el binario directamente evitando lecturas de JSON."""
        try:
            # Joblib lee directamente la ruta y maneja la descompresión
            state = joblib.load(self.model_path)
            self.model = state['model']
            
            # Soporte de retrocompatibilidad: si el dict guardado tiene 'is_trained', lo usamos
            self.is_trained = state.get('is_trained', True) 
            
            print(f"[+] Oráculo rehidratado instantáneamente desde archivo binario: {self.model_path}\n")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el oráculo binario: {e}")
            self.is_trained = False
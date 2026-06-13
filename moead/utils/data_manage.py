from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

def get_data_indices(data_dir_x: Path,
                     random_state: int = 42,
                     train_val_data: float = 0.30,
                     val_test_data: float = 0.50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la partición de datos devolviendo ÚNICAMENTE los índices.
    Garantiza Zero-Copy RAM.
    """
    if not data_dir_x.exists():
        raise FileNotFoundError(f"No se encontró el archivo .npy en {data_dir_x}")

    # Leemos solo la cabecera del archivo para saber cuántas imágenes hay (0 MB de RAM)
    X_mmap = np.load(data_dir_x, mmap_mode='r')
    total_samples = len(X_mmap)
    del X_mmap # Destruimos la referencia

    indices = np.arange(total_samples)
    
    # Partición
    idx_train, idx_rest = train_test_split(indices, test_size=train_val_data, random_state=random_state)
    idx_val, idx_test = train_test_split(idx_rest, test_size=val_test_data, random_state=random_state)

    return idx_train, idx_val, idx_test
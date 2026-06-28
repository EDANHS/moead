import json
import os

def procesar_tiempo_desde_fichero(ruta_acceso: str) -> str:
    """
    Carga un archivo JSON desde el sistema de archivos local, agrega el
    tiempo generacional y devuelve las horas totales en formato (X.Y).
    """
    # Validación previa de la existencia del elemento en el sistema de archivos
    if not os.path.exists(ruta_acceso):
        return f"Error Crítico: No se encuentra ningún archivo en la ruta especificada: '{ruta_acceso}'"
        
    try:
        # Ingesta y lectura del flujo de datos usando un manejador de contexto
        with open(ruta_acceso, 'r', encoding='utf-8') as archivo_json:
            # Deserialización directa del objeto de flujo de archivo
            estructura_datos = json.load(archivo_json)
        
        # Aislamiento del vector multidimensional de minutos
        tiempos_minutos = estructura_datos.get("generation_times_minutes", [])
        
        if not tiempos_minutos:
            return "Advertencia: El campo 'generation_times_minutes' está vacío o no existe en el JSON."
            
        # Agregación algebraica y conversión de magnitud temporal
        total_minutos = sum(tiempos_minutos)
        total_horas = total_minutos / 60.0
        
        # Formateo dimensional string (X.Y)
        return f"({total_horas:.1f})"
        
    except json.JSONDecodeError:
        return "Error Estructural: El archivo existe pero su contenido no posee un formato JSON válido."
    except PermissionError:
        return "Error de Seguridad: Permisos insuficientes para leer el archivo en la ruta provista."
    except Exception as e:
        return f"Error Inesperado durante la ejecución: {str(e)}"

# =====================================================================
# CONFIGURACIÓN DE LA RUTA DE ACCESO (Modifica este parámetro a tu gusto)
# =====================================================================
# Puedes usar rutas relativas, por ejemplo: "datos/experimento_1.json"
# O rutas absolutas: "C:/Usuarios/TuUsuario/Documentos/resultado.json"
RUTA_DEL_ARCHIVO = "data_marts/metrics_differential_evolution.json"

# Ejecución del pipeline de procesamiento
if __name__ == "__main__":
    print("Iniciando lectura y análisis del reporte temporal...")
    resultado_horas = procesar_tiempo_desde_fichero(RUTA_DEL_ARCHIVO)
    
    if "Error" in resultado_horas or "Advertencia" in resultado_horas:
        print(resultado_horas)
    else:
        print(f"Análisis Exitoso. Tiempo total de procesamiento: {resultado_horas} horas.")
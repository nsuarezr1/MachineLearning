import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def entrenar_modelo(caracteristicas, etiquetas):
    """Entrena un modelo de regresión logística."""
    X_train, X_test, y_train, y_test = train_test_split(caracteristicas, etiquetas, test_size=0.2, random_state=42)
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    precision = accuracy_score(y_test, modelo.predict(X_test))
    print(f'Precisión del modelo: {precision}')
    return modelo

def guardar_modelo(modelo, ruta_modelo):
    """Guarda un modelo entrenado en un archivo."""
    with open(ruta_modelo, 'wb') as archivo:
        pickle.dump(modelo, archivo)

def cargar_modelo(ruta_modelo):
    """Carga un modelo entrenado desde un archivo."""
    with open(ruta_modelo, 'rb') as archivo:
        modelo = pickle.load(archivo)
    return modelo

def predecir_presencia_objeto(modelo, pixeles, colores, bordes, contraste):
    """Realiza una predicción sobre la presencia de un objeto en una imagen."""
    caracteristicas = np.array([[pixeles, colores, bordes, contraste]])
    prediccion = modelo.predict(caracteristicas)
    return prediccion[0]

# Datos de ejemplo (simulados)
import numpy as np

caracteristicas = np.array([
    [0.5, 0.2, 0.8, 0.9],   # Imagen 1: objeto presente
    [0.1, 0.9, 0.3, 0.4],   # Imagen 2: objeto ausente
    [0.7, 0.4, 0.6, 0.7],   # Imagen 3: objeto presente
    [0.2, 0.8, 0.1, 0.3],   # Imagen 4: objeto ausente
    [0.9, 0.6, 0.5, 0.8],   # Imagen 5: objeto presente
    [0.3, 0.1, 0.7, 0.2],   # Imagen 6: objeto ausente
    [0.6, 0.5, 0.9, 0.1],   # Imagen 7: objeto presente
    [0.4, 0.7, 0.2, 0.6],   # Imagen 8: objeto ausente
    [0.8, 0.3, 0.4, 0.5],   # Imagen 9: objeto presente
    [0.1, 0.9, 0.6, 0.7],   # Imagen 10: objeto ausente
    [0.65, 0.35, 0.75, 0.85], # Imagen 11: objeto presente
    [0.25, 0.75, 0.45, 0.55], # Imagen 12: objeto ausente
    [0.8, 0.2, 0.9, 0.95], # Imagen 13: objeto presente
    [0.15, 0.85, 0.25, 0.35], # Imagen 14: objeto ausente
    [0.95, 0.55, 0.65, 0.75], # Imagen 15: objeto presente
    [0.35, 0.05, 0.85, 0.15], # Imagen 16: objeto ausente
    [0.7, 0.6, 0.95, 0.05], # Imagen 17: objeto presente
    [0.5, 0.8, 0.15, 0.65], # Imagen 18: objeto ausente
    [0.9, 0.4, 0.55, 0.45], # Imagen 19: objeto presente
    [0.05, 0.95, 0.7, 0.8], # Imagen 20: objeto ausente
    [0.75, 0.25, 0.85, 0.9], # imagen 21: objeto presente
    [0.2, 0.85, 0.35, 0.45], # imagen 22: objeto ausente
    [0.85, 0.45, 0.7, 0.85], # imagen 23: objeto presente
    [0.4, 0.6, 0.25, 0.3], # imagen 24: objeto ausente
    [0.9, 0.65, 0.6, 0.8], # imagen 25: objeto presente
    [0.3, 0.15, 0.75, 0.25], # imagen 26: objeto ausente
    [0.6, 0.55, 0.95, 0.1], # imagen 27: objeto presente
    [0.5, 0.75, 0.2, 0.7], # imagen 28: objeto ausente
    [0.8, 0.35, 0.4, 0.55], # imagen 29: objeto presente
    [0.15, 0.9, 0.65, 0.75] # imagen 30: objeto ausente

])

etiquetas = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) # 1: objeto presente, 0: objeto ausente
# Entrena el modelo y lo guarda
ruta_modelo = 'modelo_entrenado.pkl'
modelo = entrenar_modelo(caracteristicas, etiquetas)
guardar_modelo(modelo, ruta_modelo)

# Carga el modelo entrenado
modelo_cargado = cargar_modelo(ruta_modelo)

# Solicita al usuario que ingrese las variables
pixeles = float(input('Ingrese el valor de píxeles: '))
colores = float(input('Ingrese el valor de colores: '))
bordes = float(input('Ingrese el valor de bordes: '))
contraste = float(input('Ingrese el valor de contraste: '))

# Realiza la predicción
prediccion = predecir_presencia_objeto(modelo_cargado, pixeles, colores, bordes, contraste)

# Muestra el resultado
if prediccion == 1:
    print('La imagen contiene un objeto.')
else:
    print('La imagen no contiene un objeto.')
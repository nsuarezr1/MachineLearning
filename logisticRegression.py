import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

caracteristicas = np.array([
    [0.8, 0.7, 0.85, 0.7],  # Objeto presente: muchos pixeles, colores, alto contraste, bordes definidos
    [0.2, 0.3, 0.15, 0.2],  # Objeto ausente: pocos pixeles, colores, bajo contraste, bordes difusos
    [0.9, 0.8, 0.9, 0.8],  # Objeto presente
    [0.1, 0.2, 0.1, 0.1],  # Objeto ausente
    [0.85, 0.75, 0.8, 0.75], # Objeto presente
    [0.3, 0.4, 0.2, 0.3],  # Objeto ausente
    [0.7, 0.6, 0.75, 0.65], # Objeto presente
    [0.25, 0.35, 0.25, 0.25], # Objeto ausente
    [0.95, 0.9, 0.95, 0.9], # Objeto presente
    [0.15, 0.25, 0.05, 0.15], # Objeto ausente
    [0.8, 0.7, 0.85, 0.7],  # Objeto presente
    [0.2, 0.3, 0.15, 0.2],  # Objeto ausente
    [0.9, 0.8, 0.9, 0.8],  # Objeto presente
    [0.1, 0.2, 0.1, 0.1],  # Objeto ausente
    [0.85, 0.75, 0.8, 0.75], # Objeto presente
    [0.3, 0.4, 0.2, 0.3],  # Objeto ausente
    [0.7, 0.6, 0.75, 0.65], # Objeto presente
    [0.25, 0.35, 0.25, 0.25], # Objeto ausente
    [0.95, 0.9, 0.95, 0.9], # Objeto presente
    [0.15, 0.25, 0.05, 0.15], # Objeto ausente
    [0.8, 0.7, 0.85, 0.7],  # Objeto presente
    [0.2, 0.3, 0.15, 0.2],  # Objeto ausente
    [0.9, 0.8, 0.9, 0.8],  # Objeto presente
    [0.1, 0.2, 0.1, 0.1],  # Objeto ausente
    [0.85, 0.75, 0.8, 0.75], # Objeto presente
    [0.3, 0.4, 0.2, 0.3],  # Objeto ausente
    [0.7, 0.6, 0.75, 0.65], # Objeto presente
    [0.25, 0.35, 0.25, 0.25], # Objeto ausente
    [0.95, 0.9, 0.95, 0.9], # Objeto presente
    [0.15, 0.25, 0.05, 0.15], # Objeto ausente
    [0.8, 0.7, 0.85, 0.7],  # Objeto presente
    [0.2, 0.3, 0.15, 0.2],  # Objeto ausente
    [0.9, 0.8, 0.9, 0.8],  # Objeto presente
    [0.1, 0.2, 0.1, 0.1],  # Objeto ausente
    [0.85, 0.75, 0.8, 0.75], # Objeto presente
    [0.3, 0.4, 0.2, 0.3],  # Objeto ausente
    [0.7, 0.6, 0.75, 0.65], # Objeto presente
    [0.25, 0.35, 0.25, 0.25], # Objeto ausente
    [0.95, 0.9, 0.95, 0.9], # Objeto presente
    [0.15, 0.25, 0.05, 0.15], # Objeto ausente
    [0.8, 0.7, 0.85, 0.7],  # Objeto presente
    [0.2, 0.3, 0.15, 0.2],  # Objeto ausente
    [0.9, 0.8, 0.9, 0.8],  # Objeto presente
    [0.1, 0.2, 0.1, 0.1],  # Objeto ausente
    [0.85, 0.75, 0.8, 0.75], # Objeto presente
    [0.3, 0.4, 0.2, 0.3],  # Objeto ausente
    [0.7, 0.6, 0.75, 0.65], # Objeto presente
    [0.25, 0.35, 0.25, 0.25], # Objeto ausente
    [0.95, 0.9, 0.95, 0.9], # Objeto presente
    [0.15, 0.25, 0.05, 0.15], # Objeto ausente
    [0.8, 0.7, 0.85, 0.7],  # Objeto presente
    [0.2, 0.3, 0.15, 0.2],  # Objeto ausente
    [0.9, 0.8, 0.9, 0.8],  # Objeto presente
    [0.1, 0.2, 0.1, 0.1],  # Objeto ausente
    [0.85, 0.75, 0.8, 0.75], # Objeto presente
    [0.3, 0.4, 0.2, 0.3],  # Objeto ausente
    [0.7, 0.6, 0.75, 0.65], # Objeto presente
    [0.25, 0.35, 0.25, 0.25], # Objeto ausente
    [0.95, 0.9, 0.95, 0.9], # Objeto presente
    [0.15, 0.25, 0.05, 0.15], # Objeto ausente
    [0.8, 0.7, 0.85, 0.7],  # Objeto presente
    [0.2, 0.3, 0.15, 0.2],  # Objeto ausente
    [0.9, 0.8, 0.9, 0.8],  # Objeto presente
    [0.1, 0.2, 0.1, 0.1],  # Objeto ausente
    [0.85, 0.75, 0.8, 0.75], # Objeto presente
    [0.3, 0.4, 0.2, 0.3],  # Objeto ausente
    [0.7, 0.6, 0.75, 0.65], # Objeto presente
    [0.25, 0.35, 0.25, 0.25], # Objeto ausente
    [0.95, 0.9, 0.95, 0.9], # Objeto presente
    [0.15, 0.25, 0.05, 0.15], # Objeto ausente
    [0.8, 0.7, 0.85, 0.7],  # Objeto presente
    [0.2, 0.3, 0.15, 0.2],  # Objeto ausente
    [0.9, 0.8, 0.9, 0.8],  # Objeto presente
    [0.1, 0.2, 0.1, 0.1],  # Objeto ausente
    [0.85, 0.75, 0.8, 0.75], # Objeto presente
    [0.3, 0.4, 0.2, 0.3],  # Objeto ausente
    [0.7, 0.6, 0.75, 0.65], # Objeto presente
    [0.25, 0.35, 0.25, 0.25], # Objeto ausente
    [0.95, 0.9, 0.95, 0.9], # Objeto presente
    [0.15, 0.25, 0.05, 0.15], # Objeto ausente
    [0.8, 0.7, 0.85, 0.7],  # Objeto presente
    [0.2, 0.3, 0.15, 0.2],  # Objeto ausente
    [0.9, 0.8, 0.9, 0.8],  # Objeto presente
    [0.1, 0.2, 0.1, 0.1],  # Objeto ausente
    [0.85, 0.75, 0.8, 0.75], # Objeto presente
    [0.3, 0.4, 0.2, 0.3],  # Objeto ausente
    [0.7, 0.6, 0.75, 0.65], # Objeto presente
    [0.25, 0.35, 0.25, 0.25], # Objeto ausente
    [0.95, 0.9, 0.95, 0.9], # Objeto presente
    [0.15, 0.25, 0.05, 0.15],  # Objeto ausente
    [0.82, 0.72, 0.88, 0.78], # Objeto presente
    [0.18, 0.28, 0.12, 0.22], # Objeto ausente
    [0.87, 0.77, 0.92, 0.82], # Objeto presente
    [0.22, 0.32, 0.18, 0.25], # Objeto ausente
    [0.83, 0.73, 0.87, 0.77], # Objeto presente
    [0.19, 0.29, 0.14, 0.23], #Objeto ausente
    [0.88, 0.78, 0.93, 0.83], #Objeto presente
    [0.23, 0.33, 0.19, 0.26], #Objeto ausente
    [0.84, 0.74, 0.89, 0.79], #Objeto presente
    [0.3, 0.4, 0.21, 0.32],  # Objeto ausente
])

etiquetas = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# Crear DataFrame
df = pd.DataFrame(caracteristicas, columns=['pixeles', 'colores', 'contraste', 'bordes'])
df['objeto'] = etiquetas

# Variables independientes y dependiente
X = df[['pixeles', 'colores', 'contraste', 'bordes']]
y = df['objeto']

# División del dataset (80%/20%)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
modelo = LogisticRegression()
modelo.fit(X_entrenamiento, y_entrenamiento)

# Predicciones y métricas
predicciones = modelo.predict(X_prueba)
exactitud = accuracy_score(y_prueba, predicciones)
precision = precision_score(y_prueba, predicciones)
sensibilidad = recall_score(y_prueba, predicciones)
matriz_confusion = confusion_matrix(y_prueba, predicciones)

def predecir_nuevo(pixeles, colores, contraste, bordes):
    """Realiza la predicción para un nuevo registro."""
    entrada = np.array([[pixeles, colores, contraste, bordes]])
    return modelo.predict(entrada)[0]

def obtener_metricas():
    """Retorna las métricas del modelo."""
    return {
        'exactitud': exactitud,
        'precision': precision,
        'sensibilidad': sensibilidad,
        'matriz_confusion': matriz_confusion.tolist()
    }
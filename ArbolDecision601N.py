import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cargar el dataset desde el archivo CSV
try:
    data = pd.read_csv('CompraClientes.csv')  # Reemplaza 'tu_archivo.csv' con el nombre de tu archivo
except FileNotFoundError:
    print("Error: El archivo CSV no se encontró.")
    exit()

# Separar las características (X) de la variable objetivo (y)
X = data[['edad', 'ingreso', 'ubicacion_Rural', 'ubicacion_Suburbana', 'ubicacion_Urbana']]
y = data['compra']

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Inicializar el modelo de Árbol de Decisión
model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo con el conjunto de entrenamiento
model.fit(X_train, y_train)

print("Modelo de Árbol de Decisión entrenado.")

# Evaluar el modelo (opcional aquí, puedes hacerlo en otro script si lo prefieres)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("\nEvaluación del Modelo en el Conjunto de Prueba:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nMatriz de Confusión:")
print(confusion)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Compra', 'Compra'], yticklabels=['No Compra', 'Compra'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# Guardar el modelo usando Joblib
filename_joblib = 'modelo_arbol_decision.joblib'
joblib.dump(model, filename_joblib)

print(f"\nModelo guardado en: {filename_joblib}")
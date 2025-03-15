import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

df = pd.read_csv('TemperaturayHelados.csv')

X = df[["Temperatura (°C)"]]
y = df[["Helados vendidos (unidades)"]]

modelo = LinearRegression()
modelo.fit(X, y)

def calcularHelados(temperatura):
  
    resultado = modelo.predict(pd.DataFrame({"Temperatura (°C)": [temperatura]}))[0]
    return resultado

def generar_grafica():

    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, color='blue', label='Helados vendidos reales')
    x_range = np.linspace(min(X["Temperatura (°C)"]), max(X["Temperatura (°C)"]), 100).reshape(-1, 1)
    x_range_df = pd.DataFrame(x_range, columns=["Temperatura (°C)"])
    y_pred = modelo.predict(x_range_df)
    plt.plot(x_range, y_pred, color='red', linewidth=2, label='Línea de regresión')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Helados vendidos (unidades)')
    plt.title('Temperatura vs. Helados vendidos')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    grafica_url = base64.b64encode(img.getvalue()).decode()
    
    plt.close()
    return grafica_url


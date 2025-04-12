from flask import Flask, render_template, request
from datetime import datetime
import joblib
import pandas as pd
import regex as re
import RegresionLineal601N
import RegresionLogistica601N
from db import get_models, get_model_by_name
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()

    match_object = re.fullmatch("[a-zA-Z]+", name)
    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"
    content = f"Hello there, {clean_name} | Hour: {now}"
    return content

@app.route("/CasoUso/")
def CasoUso():
    return render_template("CasoUso.html")

@app.route("/RegresionLineal/", methods=["GET", "POST"])
def RegresionLineal():
    calculateResult = None
    plot_url = RegresionLineal601N.generar_grafica()
    if request.method == "POST":
        temperatura = float(request.form["temperatura"])
        calculateResult = RegresionLineal601N.calcularHelados(temperatura)
    return render_template("RegresionLineal.html", result=calculateResult, plot_url=plot_url)

@app.route("/CasoUsoRegresionLogistica")
def CasoUsoRegresionLogistica():
    return render_template("CasoUsoRegresionLogistica.html")


@app.route('/RegresionLogistica', methods=['GET', 'POST'])
def RegresionLogistica():
    resultado = None
    if request.method == 'POST':
        # Recuperar los datos del formulario (ahora se permiten decimales)
        pixeles = float(request.form['pixeles'])
        colores = float(request.form['colores'])
        contraste = float(request.form['contraste'])
        bordes = float(request.form['bordes'])
        # Realizar la predicción
        pred = RegresionLogistica601N.predecir_nuevo(pixeles, colores, contraste, bordes)
        resultado = "Contiene objeto" if pred == 1 else "No contiene objeto"
    metricas = RegresionLogistica601N.obtener_metricas()
    # Información del dataset para mostrar en la interfaz
    dataset_info = {
        'titulo': 'Conjunto de Datos de Imágenes',
        'objetivo': 'Determinar si una imagen contiene un objeto',
        'descripcion': 'Este conjunto de datos contiene 100 registros con las variables: pixeles, colores, contraste y bordes.'
    }
    return render_template('RegresionLogistica.html', resultado=resultado, metricas=metricas, dataset_info=dataset_info)

@app.route('/modelos')
def modelos():
    models = get_models()  
    return render_template('modelos.html', models=models)

@app.route('/modelos/<model_name>')
def model_page(model_name):
    model = get_model_by_name(model_name)
    if not model:
        return "Model Not Found", 404
    return render_template('modelo.html', model=model)

@app.route('/ArbolDecision', methods=['GET', 'POST'])
def ArbolDecision():
    resultado = None
    modelo_arbol = None  # Inicializar modelo_arbol dentro de la función
    try:
        modelo_arbol = joblib.load('modelo_arbol_decision.joblib')  # Ajusta la ruta si es necesario
        print("Modelo de Árbol de Decisión cargado exitosamente en la ruta.")
    except FileNotFoundError:
        resultado = "Error: No se encontró el archivo del modelo de Árbol de Decisión."
        print("Error: No se encontró el archivo del modelo de Árbol de Decisión en la ruta.")
    except Exception as e:
        resultado = f"Error al cargar el modelo de Árbol de Decisión: {e}"
        print(f"Error al cargar el modelo de Árbol de Decisión en la ruta: {e}")

    if request.method == 'POST':
        if modelo_arbol is None and resultado is None:
            resultado = "Error: El modelo de Árbol de Decisión no está cargado."
        elif modelo_arbol is not None:
            try:
                # Recuperar los datos del formulario
                edad = int(request.form['edad'])
                ingreso = float(request.form['ingreso'])
                ubicacion = request.form['ubicacion']

                # Crear un DataFrame con los datos ingresados y realizar one-hot encoding
                nuevo_dato = pd.DataFrame({'edad': [edad], 'ingreso': [ingreso], 'ubicacion': [ubicacion]})
                nuevo_dato_encoded = pd.get_dummies(nuevo_dato, columns=['ubicacion'], drop_first=True)

                # Asegurarse de que las columnas coincidan con las del modelo entrenado
                columnas_esperadas = ['edad', 'ingreso', 'ubicacion_Rural', 'ubicacion_Suburbana', 'ubicacion_Urbana']
                for col in columnas_esperadas:
                    if col not in nuevo_dato_encoded.columns:
                        nuevo_dato_encoded[col] = 0
                nuevo_dato_encoded = nuevo_dato_encoded[columnas_esperadas] # Asegurar el orden

                # Realizar la predicción
                prediccion = modelo_arbol.predict(nuevo_dato_encoded)[0]
                resultado = "Compra" if prediccion == 1 else "No Compra"

            except ValueError:
                resultado = "Error: Por favor, ingresa valores numéricos válidos para edad e ingreso."
            except KeyError as e:
                resultado = f"Error: Falta el campo '{e}' en el formulario."
            except Exception as e:
                resultado = f"Ocurrió un error durante la predicción: {e}"

    # Información del dataset y el modelo para mostrar en la interfaz (opcional)
    modelo_info = {
        'nombre': 'Árbol de Decisión',
        'objetivo': 'Predecir si un cliente comprará un producto',
        'variables_entrada': ['edad', 'ingreso', 'ubicacion'],
        'variable_salida': 'Compra (1) / No Compra (0)'
    }

    return render_template('ArbolDecision.html', resultado=resultado, modelo_info=modelo_info)


if __name__ == '__main__':
    app.run(debug=True)
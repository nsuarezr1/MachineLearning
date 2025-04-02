from flask import Flask, render_template, request
from datetime import datetime
import regex as re
import linearRegression601N
import logisticRegression

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello Flask"

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

@app.route("/exampleHTML/")
def exampleHTML():
    return render_template("index.html")

@app.route("/linearRegression/", methods=["GET", "POST"])
def calculateGrade():
    calculateResult = None
    plot_url = linearRegression601N.generar_grafica()
    if request.method == "POST":
        temperatura = float(request.form["temperatura"])
        calculateResult = linearRegression601N.calcularHelados(temperatura)
    return render_template("linearRegressionGrades.html", result=calculateResult, plot_url=plot_url)

@app.route("/linearLogistica/")
def logistic():
    return render_template("/linearLogistica.html")


@app.route('/regresionLogistica', methods=['GET', 'POST'])
def index():
    resultado = None
    if request.method == 'POST':
        # Recuperar los datos del formulario (ahora se permiten decimales)
        pixeles = float(request.form['pixeles'])
        colores = float(request.form['colores'])
        contraste = float(request.form['contraste'])
        bordes = float(request.form['bordes'])
        # Realizar la predicción
        pred = logisticRegression.predecir_nuevo(pixeles, colores, contraste, bordes)
        resultado = "Contiene objeto" if pred == 1 else "No contiene objeto"
    metricas = logisticRegression.obtener_metricas()
    # Información del dataset para mostrar en la interfaz
    dataset_info = {
        'titulo': 'Conjunto de Datos de Imágenes',
        'objetivo': 'Determinar si una imagen contiene un objeto',
        'descripcion': 'Este conjunto de datos contiene 100 registros con las variables: pixeles, colores, contraste y bordes.'
    }
    return render_template('logisticRegresion.html', resultado=resultado, metricas=metricas, dataset_info=dataset_info)
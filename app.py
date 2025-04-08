from flask import Flask, render_template, request
from datetime import datetime
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

if __name__ == '__main__':
    app.run(debug=True)
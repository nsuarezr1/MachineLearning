from flask import Flask, render_template, request
from datetime import datetime
import regex as re
import linearRegression601N

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

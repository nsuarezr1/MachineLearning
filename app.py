from flask import Flask, render_template
app=Flask(__name__)

@app.route("/")
def home():
    return "Hello"


@app.route("/index/")
def exampleHTML():
    return render_template("index.html")
    
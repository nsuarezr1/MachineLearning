import os
import pyodbc



def get_connection():
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={os.environ.get('DB_SERVER')};"
        f"DATABASE={os.environ.get('DB_NAME')};"
        f"UID={os.environ.get('DB_USER')};"
        f"PWD={os.environ.get('DB_PASSWORD')};"
    )
    return pyodbc.connect(conn_str)


def get_models():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, Titulo FROM modelos")  
    models = cursor.fetchall()
    conn.close()
    return models


def get_model_by_name(model_name):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT Titulo, Descripcion, referencia, urlImagen FROM modelos WHERE Titulo = ?", (model_name,))
    
    model = cursor.fetchone()
    conn.close()
    
    return model
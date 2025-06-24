import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import base64
import os

EXCEL_PATH = os.path.join("data", "Students_Social_Media_Addiction.csv")

def cargar_datos():
    df = pd.read_csv(EXCEL_PATH)
    return df.dropna()

def codificar_columnas(df, columnas):
    encoders = {}
    for col in columnas:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    return df, encoders

def graficar_modelo(x, y, y_pred, title):
    plt.figure()
    plt.scatter(x, y, color='blue', label='Real')
    plt.plot(x, y_pred, color='red', label='Predicción')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Addicted_Score')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def entrenar_modelos(target, feature):
    df = cargar_datos()
    df, encoders = codificar_columnas(df, [target, feature])

    X = df[[feature]]
    y = df[target]

    resultados = []

    # Regresión Lineal
    try:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        score = mean_squared_error(y, y_pred)
        graf = graficar_modelo(X, y, y_pred, f"Regresión Lineal: {feature}")
        resultados.append({
            "modelo": "regresión lineal",
            "score": round(score, 4),
            "grafico": graf
        })
    except Exception as e:
        resultados.append({"modelo": "regresión lineal", "error": str(e)})

    # Regresión Logística (clasificación del score)
    try:
        y_bin = (y > y.median()).astype(int)
        model = LogisticRegression()
        model.fit(X, y_bin)
        pred = model.predict(X)
        score = accuracy_score(y_bin, pred)
        resultados.append({
            "modelo": "regresión logística",
            "score": round(score, 4)
        })
    except Exception as e:
        resultados.append({"modelo": "regresión logística", "error": str(e)})

    # Árbol de decisión
    try:
        y_bin = (y > y.median()).astype(int)
        model = DecisionTreeClassifier()
        model.fit(X, y_bin)
        pred = model.predict(X)
        score = accuracy_score(y_bin, pred)
        resultados.append({
            "modelo": "árbol de decisión",
            "score": round(score, 4)
        })
    except Exception as e:
        resultados.append({"modelo": "árbol de decisión", "error": str(e)})

    # Clustering (KMeans)
    try:
        model = KMeans(n_clusters=3, n_init='auto', random_state=42)
        model.fit(X)
        cluster_pred = model.predict(X)
        plt.figure()
        plt.scatter(X, y, c=cluster_pred, cmap='viridis')
        plt.title(f"KMeans Clustering: {feature}")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        graf = base64.b64encode(buf.read()).decode('utf-8')
        resultados.append({
            "modelo": "kmeans",
            "grafico": graf
        })
    except Exception as e:
        resultados.append({"modelo": "kmeans", "error": str(e)})

    return resultados

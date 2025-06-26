# model_utils_excel.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Carpeta donde están los excels
DATA_FOLDER = 'data/excel_cargado'


def cargar_ultimo_excel():
    archivos = sorted(os.listdir(DATA_FOLDER), reverse=True)
    for archivo in archivos:
        if archivo.endswith('.csv'):
            return pd.read_csv(os.path.join(DATA_FOLDER, archivo))
        elif archivo.endswith('.xlsx') or archivo.endswith('.xls'):
            return pd.read_excel(os.path.join(DATA_FOLDER, archivo))
    raise FileNotFoundError("No se encontró un archivo válido")


def codificar_columnas(df):
    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    return df, le_dict


def graficar_modelo(X, y, modelo, nombre):
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue')
    if hasattr(modelo, 'predict'):
        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = modelo.predict(x_line)
        ax.plot(x_line, y_pred, color='red')
    ax.set_title(nombre)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


def entrenar_modelos_desde_lista(combinaciones):
    df = cargar_ultimo_excel()
    df.dropna(inplace=True)
    df, codificadores = codificar_columnas(df)

    resultados = []

    for target, features in combinaciones:
        if isinstance(features, str):
            features = [features]

        try:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            resultado = {"target": target, "features": features, "modelos": []}

            # Regresión Lineal
            modelo_rl = LinearRegression()
            modelo_rl.fit(X_train, y_train)
            score_rl = modelo_rl.score(X_test, y_test)
            grafica_rl = None
            if len(features) == 1:
                grafica_rl = graficar_modelo(X[features[0]].values.reshape(-1, 1), y, modelo_rl, f"Regresión Lineal: {target}~{features[0]}")
            resultado["modelos"].append({"tipo": "regresion_lineal", "score": score_rl, "grafica": grafica_rl})

            # Regresión Logística 
            if len(set(y)) <= 10:
                modelo_log = LogisticRegression(max_iter=1000)
                modelo_log.fit(X_train, y_train)
                score_log = modelo_log.score(X_test, y_test)
                resultado["modelos"].append({"tipo": "regresion_logistica", "score": score_log})

            # Árbol de decisión
            modelo_arbol = DecisionTreeClassifier()
            modelo_arbol.fit(X_train, y_train)
            score_arbol = modelo_arbol.score(X_test, y_test)
            resultado["modelos"].append({"tipo": "arbol_decision", "score": score_arbol})

            # KMeans clustering
            if len(set(y)) > 2:  # solo si el target tiene más de 2 clases
                modelo_kmeans = KMeans(n_clusters=3, n_init=10)
                modelo_kmeans.fit(X)
                resultado["modelos"].append({"tipo": "clustering", "labels": modelo_kmeans.labels_.tolist()})

            resultados.append(resultado)
        except Exception as e:
            resultados.append({"target": target, "features": features, "error": str(e)})

    return resultados

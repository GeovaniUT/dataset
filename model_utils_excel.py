# model_utils_excel.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from predictors.prediccion_cols_vacias import prediccion_cols_vacias

# Carpeta donde están los excels
DATA_FOLDER = 'data/excel_cargado'


def cargar_excel():
    archivos = sorted(os.listdir(DATA_FOLDER))
    for archivo in archivos:
        if archivo.endswith('.csv'):
            print("Ruta de archivo csv", os.path.join(DATA_FOLDER, archivo))
            return pd.read_csv(os.path.join(DATA_FOLDER, archivo))

        elif archivo.endswith('.xlsx') or archivo.endswith('.xls'):
            print("Ruta de archivo xlsx", os.path.join(DATA_FOLDER, archivo))
            return pd.read_excel(os.path.join(DATA_FOLDER, archivo))
    raise FileNotFoundError("No se encontró un archivo válido")

def cargar_excel_enriquecido():
    DATA_FOLDER = 'data/excel_cargado'  # Ruta específica
    ARCHIVO_ESPERADO = 'excel_enriquecido.xlsx'  # Nombre exacto del archivo
    
    ruta_completa = os.path.join(DATA_FOLDER, ARCHIVO_ESPERADO)
    
    if not os.path.exists(ruta_completa):
        raise FileNotFoundError(
            f"No se encontró el archivo enriquecido en: {ruta_completa}\n"
            f"Archivos disponibles en {DATA_FOLDER}: {os.listdir(DATA_FOLDER)}"
        )
    
    print(f"Cargando archivo enriquecido desde: {ruta_completa}")
    return pd.read_excel(ruta_completa)

#Función para mantener persistencia de datos generados por entrenamiento
def guardar_modelo(modelo, target, tipo_modelo, ruta_base="modelos"):
    """Guarda un modelo en disco y devuelve su ruta"""
    ruta_target = os.path.join(ruta_base, target)
    os.makedirs(ruta_target, exist_ok=True)
    
    nombre_archivo = f"{tipo_modelo}.pkl"
    ruta_completa = os.path.join(ruta_target, nombre_archivo)
    
    joblib.dump(modelo, ruta_completa)
    return ruta_completa

def guardar_codificadores(codificadores, ruta="modelos/codificadores"):
    """Guarda todos los codificadores usados en el preprocesamiento"""
    os.makedirs(ruta, exist_ok=True)
    for nombre, codificador in codificadores.items():
        joblib.dump(codificador, os.path.join(ruta, f"{nombre}_encoder.pkl"))

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

    df_crudo= cargar_excel()

    #Validación de existencia de archivo de excelEnriquecido:
    ruta_excel = 'data/excel_cargado/excel_enriquecido.xlsx'
    if not os.path.exists(ruta_excel):
            print("Archivo enriquecido no encontrado. Generando datos...")
            prediccion_cols_vacias(df_crudo)  # Ejecutar función que genera el archivo
            print("Datos enriquecidos generados correctamente")

    df = cargar_excel_enriquecido()
    print("Columnas de archivo:", list(df.columns.values))
    print("Total de registros en archivo:", len(df))
    df.dropna(inplace=True)  # Cambiado a True para eliminar filas con NaN
    
    # Codificar columnas categóricas y guardar codificadores
    df, codificadores = codificar_columnas(df)
    guardar_codificadores(codificadores)
    
    resultados = []

    for target, features in combinaciones:
        if isinstance(features, str):
            features = [features]

        try:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            resultado = {
                "target": target,
                "features": features,
                "modelos": [],
                "modelos_persistidos": []
            }

            # --- Regresión Lineal ---
            modelo_rl = LinearRegression()
            modelo_rl.fit(X_train, y_train)
            score_rl = modelo_rl.score(X_test, y_test)
            
            # Persistencia
            ruta_modelo_rl = guardar_modelo(
                modelo_rl, 
                target, 
                "regresion_lineal"
            )
            
            # Gráfica (solo para una feature)
            grafica_rl = None
            if len(features) == 1:
                grafica_rl = graficar_modelo(
                    X[features[0]].values.reshape(-1, 1), 
                    y, 
                    modelo_rl, 
                    f"Regresión Lineal: {target}~{features[0]}"
                )
            
            resultado["modelos"].append({
                "tipo": "regresion_lineal",
                "score": score_rl,
                "grafica": grafica_rl
            })
            resultado["modelos_persistidos"].append(ruta_modelo_rl)

            # --- Regresión Logística ---
            if len(y.unique()) <= 10:  # Para clasificación
                modelo_log = LogisticRegression(max_iter=1000)
                modelo_log.fit(X_train, y_train)
                score_log = modelo_log.score(X_test, y_test)
                
                ruta_modelo_log = guardar_modelo(
                    modelo_log,
                    target,
                    "regresion_logistica"
                )
                
                resultado["modelos"].append({
                    "tipo": "regresion_logistica",
                    "score": score_log
                })
                resultado["modelos_persistidos"].append(ruta_modelo_log)

            # --- Árbol de Decisión ---
            modelo_arbol = DecisionTreeClassifier()
            modelo_arbol.fit(X_train, y_train)
            score_arbol = modelo_arbol.score(X_test, y_test)
            
            ruta_modelo_arbol = guardar_modelo(
                modelo_arbol,
                target,
                "arbol_decision"
            )
            
            resultado["modelos"].append({
                "tipo": "arbol_decision",
                "score": score_arbol
            })
            resultado["modelos_persistidos"].append(ruta_modelo_arbol)

            # --- KMeans Clustering ---
            if len(y.unique()) > 2:  # Para múltiples clases
                modelo_kmeans = KMeans(n_clusters=3, n_init=10)
                modelo_kmeans.fit(X)
                
                ruta_modelo_kmeans = guardar_modelo(
                    modelo_kmeans,
                    target,
                    "kmeans_clustering"
                )
                
                resultado["modelos"].append({
                    "tipo": "clustering",
                    "labels": modelo_kmeans.labels_.tolist()
                })
                resultado["modelos_persistidos"].append(ruta_modelo_kmeans)

            resultados.append(resultado)

        except Exception as e:
            resultados.append({
                "target": target,
                "features": features,
                "error": str(e)
            })

    return resultados


                

                
                
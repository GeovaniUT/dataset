from flask import Blueprint, jsonify
from model_utils_excel import cargar_excel_enriquecido, codificar_columnas
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import joblib
from pathlib import Path
from flask import jsonify, request  
import pandas as pd
import io
from predictors.adiction_grafic import predecir_adiccion_porcentual

viz_blueprint = Blueprint('viz_routes', __name__)

# Helper para convertir gráficas a base64
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# 1. Gráfica de Regresión Lineal
@viz_blueprint.route('/grafica-regresion/<target>/<feature>', methods=['GET'])
def grafica_regresion(target, feature):
    try:
        df = cargar_excel_enriquecido()
        df, _ = codificar_columnas(df)
        
        X = df[[feature]].values
        y = df[target].values
        
        # Entrenar modelo (o reutilizar si ya está entrenado)
        model = LinearRegression()
        model.fit(X, y)
        
        # Generar gráfica
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, y, color='blue', label='Datos reales')
        ax.plot(X, model.predict(X), color='red', linewidth=2, label='Predicción')
        ax.set_title(f'Regresión Lineal: {target} ~ {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.legend()
        
        return jsonify({
            "grafica": plot_to_base64(fig),
            "r2_score": model.score(X, y)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 2. Gráfica de Árbol de Decisión (Importancia de Features)
@viz_blueprint.route('/grafica-arbol/<target>', methods=['GET'])
def grafica_arbol(target):
    try:
        df = cargar_excel_enriquecido()
        df, _ = codificar_columnas(df)
        
        X = df.drop(columns=[target]).select_dtypes(include='number')
        y = df[target].values
        
        model = DecisionTreeClassifier()
        model.fit(X, y)
        
        # Gráfica de importancia
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(X.columns, model.feature_importances_)
        ax.set_title(f'Importancia de Features (Árbol para {target})')
        
        return jsonify({
            "grafica": plot_to_base64(fig),
            "importancias": dict(zip(X.columns, model.feature_importances_.round(4)))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3. Gráfica de Clustering (K-Means)
@viz_blueprint.route('/grafica-clustering/<feature1>/<feature2>', methods=['GET'])
def grafica_clustering(feature1, feature2):
    try:
        df = cargar_excel_enriquecido()
        df, _ = codificar_columnas(df)
        
        X = df[[feature1, feature2]].values
        
        model = KMeans(n_clusters=3, n_init=10)
        clusters = model.fit_predict(X)
        
        # Gráfica de clusters
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        ax.set_title(f'Clustering: {feature1} vs {feature2}')
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        fig.colorbar(scatter, label='Cluster')
        
        return jsonify({
            "grafica": plot_to_base64(fig),
            "centroides": model.cluster_centers_.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
from flask import Blueprint, jsonify, request
import joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

viz_blueprint = Blueprint('viz', __name__)

@viz_blueprint.route('/predict-with-plot', methods=['GET'])
def predict_with_plot():
    """
    Endpoint que recibe 2 valores numéricos (age, usage_hours),
    devuelve predicción + gráfica de la relación.
    Ejemplo: /predict-with-plot?age=25&usage_hours=3
    """
    try:
        # Validar parámetros
        age = float(request.args.get('age'))
        usage_hours = float(request.args.get('usage_hours'))
    except (TypeError, ValueError):
        return jsonify({"error": "Los parámetros 'age' y 'usage_hours' deben ser números"}), 400

    try:
        # Cargar modelo y datos históricos
        model = joblib.load("models/addicted_score_model.pkl")
        df = pd.read_excel('data/excel_cargado/excel_enriquecido.xlsx')
        
        # Generar predicción
        input_data = np.array([[age, usage_hours]])
        prediction = model.predict(input_data)[0]

        # --- Crear gráfica ---
        plt.figure(figsize=(10, 6))
        
        # 1. Puntos históricos
        plt.scatter(
            df['Age'], 
            df['Avg_Daily_Usage_Hours'], 
            c=df['addicted_score'], 
            cmap='viridis', 
            label='Datos históricos'
        )
        
        # 2. Destacar el input del usuario
        plt.scatter(
            [age], 
            [usage_hours], 
            c='red', 
            s=200, 
            marker='X', 
            label=f'Tu input (Predicción: {prediction:.1f})'
        )
        
        plt.colorbar(label='Nivel de adicción')
        plt.xlabel('Edad (Age)')
        plt.ylabel('Horas de uso diario (Avg_Daily_Usage_Hours)')
        plt.title('Relación: Edad vs Horas de Uso (Color = Adicción)')
        plt.legend()

        # Convertir gráfica a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return jsonify({
            "prediction": round(float(prediction), 2),
            "plot": f"data:image/png;base64,{plot_base64}",
            "input_data": {
                "age": age,
                "usage_hours": usage_hours
            },
            "model_metadata": {
                "features_used": ["Age", "Avg_Daily_Usage_Hours"],
                "model_type": type(model).__name__
            }
        })

    except FileNotFoundError:
        return jsonify({"error": "Modelo o datos no encontrados"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
from flask import request

@viz_blueprint.route('/grafica-adiccion/<path:horas_uso>/<path:horas_sueno>', methods=['GET'])
def grafica_adiccion(horas_uso, horas_sueno):
    try:
        # Convertir a float (acepta tanto enteros como decimales)
        horas_uso = float(horas_uso)
        horas_sueno = float(horas_sueno)
        
        resultado = predecir_adiccion_porcentual(horas_uso, horas_sueno)
        return jsonify(resultado)
        
    except ValueError:
        return jsonify({"error": "Los parámetros deben ser números (ej: 2 o 2.5)"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
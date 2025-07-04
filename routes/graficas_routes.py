# Configurar matplotlib para usar backend no interactivo
import matplotlib
matplotlib.use('Agg')

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
from predictors.academic_grafic import predecir_afectacion_academica
from predictors.mental_health import predecir_salud_mental


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
    
@viz_blueprint.route('/prediccion-rendimiento/<path:horas_uso>/<path:horas_sueno>', methods=['GET'])
def predecir_rendimiento(horas_uso: float, horas_sueno: float):
   
    try:
        # Llamar a la función de predicción
        resultado = predecir_afectacion_academica(horas_uso, horas_sueno)
        
        # Estructurar respuesta
        response = {
            "prediccion_booleana": resultado["prediccion"] == "Sí",
            "probabilidad_afectacion": resultado["probabilidad"],
            "mensaje": resultado["mensaje"],
            "valores_ingresados": {
                "horas_diarias_uso": horas_uso,
                "horas_sueño_nocturno": horas_sueno
            },
            "modelo_metadata": resultado["model_metadata"]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": "Error en la predicción",
            "detalles": str(e)
        }), 500

@viz_blueprint.route('/grafica-salud-mental/<float:horas_sueno>/<int:estatus_relacion>', methods=['GET'])
def grafica_salud_mental(horas_sueno, estatus_relacion):
    try:
        # Importar función de generación de datos
        from utils.data_helpers.metal_health_data_helper import generar_datos_salud_mental
        from sklearn.ensemble import RandomForestRegressor
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Generar datos sintéticos
        df_sintetico = generar_datos_salud_mental(8000)

        # 2. Entrenar modelo
        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=6,
            min_samples_split=8,
            random_state=42
        )
        model.fit(
            df_sintetico[['sleep_hours_per_night', 'relationship_status']],
            df_sintetico['mental_health_score']
        )

        # 3. Predecir para el usuario
        X_input = np.array([[horas_sueno, estatus_relacion]])
        pred = model.predict(X_input)[0]

        # 4. Generar gráfica: salud mental vs horas de sueño
        horas_sueño_test = np.linspace(2, 10, 20)
        predicciones = model.predict(
            np.column_stack((horas_sueño_test, [estatus_relacion] * len(horas_sueño_test)))
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(horas_sueño_test, predicciones, label="Predicción Salud Mental")
        ax.axvline(horas_sueno, color='red', linestyle='--', label="Tus horas de sueño")
        ax.set_xlabel("Horas de sueño por noche")
        ax.set_ylabel("Puntaje de salud mental")
        ax.set_title("Relación entre sueño y salud mental")
        ax.legend()
        ax.grid(True)

        # 5. Convertir imagen a base64
        grafica_base64 = plot_to_base64(fig)

        # 6. Interpretación
        if pred < 4:
            mensaje = "⚠️ Salud mental baja, se recomienda apoyo."
        elif pred < 7:
            mensaje = "😐 Salud mental promedio."
        else:
            mensaje = "😊 Salud mental positiva."

        return jsonify({
            "salud_mental_score": round(float(pred), 2),
            "mensaje": mensaje,
            "grafica_base64": grafica_base64,
            "valores_ingresados": {
                "horas_sueno": horas_sueno,
                "estatus_relacion": estatus_relacion
            },
            "modelo_metadata": {
                "modelo_usado": "RandomForestRegressor",
                "features": ["sleep_hours_per_night", "relationship_status"],
                "modelo_entrenado_en_runtime": True
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

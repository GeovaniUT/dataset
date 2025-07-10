# Configurar matplotlib para usar backend no interactivo
import matplotlib
matplotlib.use('Agg')

from flask import Blueprint, jsonify
from model_utils_excel import cargar_pkl_enriquecido, codificar_columnas
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
from predictors.SleepQuality import PredecirSleepQuality
from predictors.ConflictRisk import PredecirConflictRisk
from predictors.RecommendedScreenTime import PredecirRecommendedScreenTime
from predictors.SocialWellbeingScore import PredecirSocialWellbeingScore
from predictors.StudyEfficiencyScore import PredecirStudyEfficiencyScore


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
        df = cargar_pkl_enriquecido()
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
        df = cargar_pkl_enriquecido()
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
        df = cargar_pkl_enriquecido()
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

@viz_blueprint.route('/grafica-salud-mental/<path:horas_sueno>/<path:estatus_relacion>', methods=['GET'])
def grafica_salud_mental(horas_sueno, estatus_relacion):
    try:
        from utils.data_helpers.metal_health_data_helper import generar_datos_salud_mental
        from sklearn.ensemble import RandomForestRegressor
        import matplotlib.pyplot as plt
        import numpy as np

        # Generar datos sintéticos para el modelo
        df = generar_datos_salud_mental(8000)

        # Entrenar el modelo
        modelo = RandomForestRegressor(
            n_estimators=120,
            max_depth=6,
            min_samples_split=8,
            random_state=42
        )
        modelo.fit(
            df[['sleep_hours_per_night', 'relationship_status']],
            df['mental_health_score']
        )

        # Convertir parámetros de entrada
        horas_sueno = float(horas_sueno)
        estatus_relacion = int(estatus_relacion)

        # Predecir salud mental con los datos del usuario
        entrada = np.array([[horas_sueno, estatus_relacion]])
        prediccion = modelo.predict(entrada)[0]

        # Generar gráfica de predicción en función de las horas de sueño
        rango_sueno = np.linspace(2, 10, 20)
        predicciones = modelo.predict(
            np.column_stack((rango_sueno, [estatus_relacion] * len(rango_sueno)))
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        # Línea de predicción
        ax.plot(rango_sueno, predicciones, label="Predicción de salud mental", color='blue')

        # Línea vertical con las horas que el usuario duerme
        ax.axvline(horas_sueno, color='red', linestyle='--', label="Tus horas de sueño")

        # Etiquetas descriptivas en ejes
        ax.set_xlabel("Horas de sueño por noche (1 a 10)")
        ax.set_ylabel("Puntaje de salud mental (0 a 10)")
        ax.set_title("Relación entre sueño y salud mental\n(según tu estado de relación)")
        ax.set_ylim(0, 10)
        ax.grid(True)
        ax.legend() 

        # Agregar texto descriptivo al lado derecho del eje Y
        ax.text(10.5, 1.5, "😟 Bajo (0–3.9)", color='red', fontsize=10, va='center')
        ax.text(10.5, 5.5, "😐 Promedio (4–6.9)", color='orange', fontsize=10, va='center')
        ax.text(10.5, 8.5, "😊 Positiva (7–10)", color='green', fontsize=10, va='center')



        # Convertir imagen a base64
        grafica_base64 = plot_to_base64(fig)

        # Interpretar resultado
        if prediccion < 4:
            mensaje = "⚠️ Salud mental baja, se recomienda apoyo."
        elif prediccion < 7:
            mensaje = "😐 Salud mental promedio."
        else:
            mensaje = "😊 Salud mental positiva."

        return jsonify({
            "salud_mental_score": round(float(prediccion), 2),
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

@viz_blueprint.route('/grafica-sleep-quality/<float:horas_sueno>/<int:mental_health>/<int:addicted_score>', methods=['GET'])
def grafica_sleep_quality(horas_sueno, mental_health, addicted_score):
    try:
        Resultado = PredecirSleepQuality(horas_sueno, mental_health, addicted_score)
        return jsonify(Resultado)
    except Exception as E:
        return jsonify({"error": str(E)}), 500

@viz_blueprint.route('/grafica-conflict-risk/<int:addicted_score>/<float:avg_usage>/<int:relationship_status>/<platform>', methods=['GET'])
def grafica_conflict_risk(addicted_score, avg_usage, relationship_status, platform):
    # Mapeo de nombre de plataforma a entero
    plataforma_map = {
        "Facebook": 1,
        "Instagram": 2,
        "Twitter": 3,
        "TikTok": 4,
        "YouTube": 5,
        "LinkedIn": 6,
        "Snapchat": 7,
        "WhatsApp": 8,
        "Otra": 9,
    }
    try:
        # Si ya es un número, usarlo directamente
        platform_int = int(platform)
    except ValueError:
        # Convertir texto a entero vía diccionario, por defecto 9 (Otra)
        platform_int = plataforma_map.get(platform, 9)
    try:
        resultado = PredecirConflictRisk(addicted_score, avg_usage, relationship_status, platform_int)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@viz_blueprint.route('/grafica-recommended-screen-time/<int:age>/<int:addicted_score>/<int:mental_health>', methods=['GET'])
def grafica_recommended_screen_time(age, addicted_score, mental_health):
    try:
        resultado = PredecirRecommendedScreenTime(age, addicted_score, mental_health)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@viz_blueprint.route('/grafica-social-wellbeing/<int:relationship_status>/<int:mental_health>/<int:platform>/<int:addicted_score>', methods=['GET'])
def grafica_social_wellbeing(relationship_status, mental_health, platform, addicted_score):
    try:
        resultado = PredecirSocialWellbeingScore(relationship_status, mental_health, platform, addicted_score)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@viz_blueprint.route('/grafica-study-efficiency/<int:affects_academic>/<float:avg_usage>/<float:sleep_hours>', methods=['GET'])
def grafica_study_efficiency(affects_academic, avg_usage, sleep_hours):
    try:
        resultado = PredecirStudyEfficiencyScore(affects_academic, avg_usage, sleep_hours)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

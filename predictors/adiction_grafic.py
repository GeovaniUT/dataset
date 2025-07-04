import matplotlib.pyplot as plt
import base64
import io
import numpy as np
from flask import jsonify
from utils.data_helpers.adiction_data_helper import generar_datos_con_rf_col_adic
from sklearn.ensemble import RandomForestRegressor
import itertools

# Mensajes de recomendación (se rotarán secuencialmente)
mensajes_alto_riesgo = [
    "¡Tu uso de redes es alto! Prueba establecer horarios sin pantallas antes de dormir 💤",
    "¿Sabías que reducir 1 hora diaria de redes puede mejorar tu productividad? 📱⏱️",
    "Intenta actividades offline: leer, pasear o meditar pueden ser grandes alternativas 🌿",
    "Recomendación: Desactiva notificaciones para evitar distracciones constantes 🔕"
]

mensajes_bajo_riesgo = [
    "¡Buen trabajo! Mantienes un equilibrio saludable con la tecnología 👏",
    "Sigue así: tu uso moderado de redes es un gran ejemplo 📱✅",
    "Felicidades por priorizar tu bienestar digital 🌟",
    "Tu balance entre vida digital y real es inspirador 😊"
]

contador_mensajes = itertools.cycle(range(4))


def predecir_adiccion_porcentual(usage_hours, sleep_hours):
    """
    Recibe valores numéricos y devuelve:
    - Predicción en porcentaje (0-100%)
    - Gráfica en base64
    - Datos de entrada
    """
    # Cargar datos sintéticos (como en tu función original)
    df_sintetico = generar_datos_con_rf_col_adic(5000)
    
    # Entrenar modelo (RandomForest)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df_sintetico[['avg_daily_usage_hours', 'sleep_hours_per_night']]
    y = df_sintetico['addicted_score']
    model.fit(X, y)
    
    # Predecir score (1-10) y convertir a porcentaje (0-100%)
    score = model.predict([[usage_hours, sleep_hours]])[0]
    porcentaje_adiccion = (score / 10) * 100  # Suponiendo que el score máximo es 10
    porcentaje_redondeado = round(porcentaje_adiccion, 2)

     # Seleccionar mensaje rotativo según el riesgo
    indice = next(contador_mensajes)
    if porcentaje_redondeado > 65:
        mensaje = mensajes_alto_riesgo[indice]
        nivel_riesgo = "alto"
    else:
        mensaje = mensajes_bajo_riesgo[indice]
        nivel_riesgo = "bajo"


    # --- Generar gráfica ---
    plt.figure(figsize=(10, 6))
    
    # Gráfica de dispersión histórica
    plt.scatter(
        df_sintetico['avg_daily_usage_hours'], 
        df_sintetico['sleep_hours_per_night'], 
        c=df_sintetico['addicted_score'], 
        cmap='RdYlGn_r',  # Rojo (alto) -> Verde (bajo)
        alpha=0.6
    )
    
    # Destacar input del usuario
    plt.scatter(
        [usage_hours], 
        [sleep_hours], 
        s=200, 
        c='black', 
        marker='X', 
        label=f'Tu uso: {porcentaje_adiccion:.1f}% adicción'
    )
    
    plt.colorbar(label='Nivel de adicción (1-10)')
    plt.xlabel('Horas diarias de uso')
    plt.ylabel('Horas de sueño nocturno')
    plt.title('Relación: Uso vs Sueño (Entre más rojo, mayor adicción)')
    plt.legend()

    # Convertir gráfica a base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=90)
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return {
        "prediccion_porcentaje": porcentaje_redondeado,
        "nivel_riesgo": nivel_riesgo,
        "mensaje": mensaje,
        "grafica": f"data:image/png;base64,{plot_base64}",
        "valores_ingresados": {
            "horas_diarias_uso": usage_hours,
            "horas_sueño_nocturno": sleep_hours
        },
        "modelo_metadata": {
            "algoritmo": "RandomForestRegressor",
            "variables": ["avg_daily_usage_hours", "sleep_hours_per_night"]
        }
    }
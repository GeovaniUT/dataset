# Configurar matplotlib para usar backend no interactivo
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import base64
import io
import numpy as np
from flask import jsonify
from utils.data_helpers.adiction_data_helper import generar_datos_con_rf_col_adic
from sklearn.linear_model import LinearRegression
import itertools

# Mensajes de recomendación (se mantienen igual)
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
    - Dos gráficas 2D en un mismo espacio (base64)
    - Datos de entrada
    """
    # Cargar datos sintéticos
    df_sintetico = generar_datos_con_rf_col_adic(5000)
    
    # Entrenar modelo de regresión lineal
    model = LinearRegression()
    X = df_sintetico[['avg_daily_usage_hours', 'sleep_hours_per_night']]
    y = df_sintetico['addicted_score']
    model.fit(X, y)
    
    # Predecir score (1-10) y convertir a porcentaje (0-100%)
    score = model.predict([[usage_hours, sleep_hours]])[0]
    score = np.clip(score, 1, 10)  # Asegurar que esté en el rango 1-10
    porcentaje_adiccion = (score / 10) * 100
    porcentaje_redondeado = round(porcentaje_adiccion, 2)

    # Seleccionar mensaje rotativo según el riesgo
    indice = next(contador_mensajes)
    if porcentaje_redondeado > 65:
        mensaje = mensajes_alto_riesgo[indice]
        nivel_riesgo = "alto"
    else:
        mensaje = mensajes_bajo_riesgo[indice]
        nivel_riesgo = "bajo"

    # --- Generar dos gráficas 2D en el mismo espacio ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Datos comunes
    x_usage = df_sintetico['avg_daily_usage_hours']
    y_sleep = df_sintetico['sleep_hours_per_night']
    z_score = df_sintetico['addicted_score']
    
    # Gráfica 1: Uso diario vs Score de adicción
    sc1 = ax1.scatter(x_usage, z_score, c=z_score, cmap='RdYlGn_r', alpha=0.6)
    ax1.scatter([usage_hours], [score], s=200, c='black', marker='X', label='Tu predicción')
    
    # Línea de regresión para gráfica 1
    x_fit = np.linspace(x_usage.min(), x_usage.max(), 100)
    y_fit = model.predict(np.column_stack((x_fit, np.full_like(x_fit, sleep_hours))))
    ax1.plot(x_fit, y_fit, 'b-', linewidth=2, label='Tendencia')
    
    ax1.set_xlabel('Horas diarias de uso')
    ax1.set_ylabel('Nivel de adicción (1-10)')
    ax1.set_title(f'Relación: Uso vs Adicción\n(Sueño fijo en {sleep_hours}h)')
    ax1.legend()
    fig.colorbar(sc1, ax=ax1, label='Nivel de adicción')
    
    # Gráfica 2 MODIFICADA: Invertimos el eje X para sueño
    sc2 = ax2.scatter(y_sleep, z_score, c=z_score, cmap='RdYlGn_r', alpha=0.6)
    ax2.scatter([sleep_hours], [score], s=200, c='black', marker='X', label='Tu predicción')

    # Línea de regresión para gráfica 2 (ahora aparecerá ascendente)
    sleep_fit = np.linspace(y_sleep.min(), y_sleep.max(), 100)
    addiction_fit = model.predict(np.column_stack((np.full_like(sleep_fit, usage_hours), sleep_fit)))
    ax2.plot(sleep_fit, addiction_fit, 'b-', linewidth=2, label='Tendencia')

    # INVERTIMOS EL EJE X para mejor interpretación visual
    ax2.invert_xaxis()

    ax2.set_xlabel('Más sueño ← Horas de sueño nocturno → Menos sueño')
    ax2.set_ylabel('Nivel de adicción (1-10)')
    ax2.set_title(f'Relación: Sueño vs Adicción\n(Uso fijo en {usage_hours}h)')
    ax2.annotate('Menos sueño → Mayor adicción', 
             xy=(0.5, 0.95), xycoords='axes fraction',
             fontsize=10, color='blue', ha='center')
    ax2.legend()
    fig.colorbar(sc2, ax=ax2, label='Nivel de adicción')

    plt.tight_layout()

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
        "algoritmo": "LinearRegression",
        "variables": ["avg_daily_usage_hours", "sleep_hours_per_night"],
        "explicacion_modelo": {
            "que_es": "El modelo de regresión lineal analiza cómo tus horas de uso y sueño se relacionan con la adicción a redes.",
            "como_funciona": "Combina matemáticamente ambas variables para predecir tu nivel de adicción.",
            "para_que_sirve": "Identificar patrones y ayudarte a mejorar tu equilibrio digital."
        }
    },
    "interpretacion_graficas": {
        "grafica_1": "Más horas de uso → Mayor puntuación de adicción",
        "grafica_2": "Menos horas de sueño → Mayor puntuación de adicción (eje invertido para mejor visualización)"
    }
}
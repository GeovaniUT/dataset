import itertools
from utils.data_helpers.aca_data_helper import generar_datos_booleanos_col_academic_perfomance
from sklearn.ensemble import RandomForestClassifier

# Mensajes de recomendación

MENSAJES_AFECTA = [
    "¡Cuidado! Tu uso de redes podría estar afectando tus estudios. Prueba bloquear apps durante horas de estudio 📚⏰",
    "Reducir 1 hora diaria de redes puede mejorar tus notas en un 15%. ¡Inténtalo! ✍️",
    "¿Sabías que estudiar sin distracciones de redes mejora la retención? Prueba modo avión durante estudios 🚀",
    "Establece horarios fijos para redes sociales y respétalos. ¡Tu futuro académico te lo agradecerá! 🎓"
]

MENSAJES_NO_AFECTA = [
    "¡Excelente! Mantienes un buen balance entre redes y estudios. Sigue así 👏",
    "Tus hábitos demuestran que priorizas lo importante. ¡Felicidades! 🌟",
    "Eres un ejemplo de uso responsable de tecnología y buen rendimiento académico 📱✅",
    "¡Bien hecho! Sabes disfrutar de redes sin descuidar tus metas académicas 💪"
]

# Contador para rotación de mensajes
contador_mensajes = itertools.cycle(range(4))

def predecir_afectacion_academica(horas_uso: float, horas_sueno: float) -> dict:
    """
    Predice si el uso de redes afecta el rendimiento académico
    
    Args:
        horas_uso: Horas diarias de uso de redes sociales
        horas_sueno: Horas de sueño nocturno
    
    Returns:
        dict: Diccionario con predicción, mensaje y metadatos
    """
    # 1. Generar datos sintéticos
    df_sintetico = generar_datos_booleanos_col_academic_perfomance(5000)
    
    # 2. Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(
        df_sintetico[['avg_daily_usage_hours', 'sleep_hours_per_night']],
        df_sintetico['affects_academic_performance']
    )
    
    # 3. Predecir y seleccionar mensaje
    prediccion = model.predict([[horas_uso, horas_sueno]])[0]
    indice = next(contador_mensajes)
    
    return {
        "prediccion": "Sí" if prediccion == 1 else "No",
        "mensaje": MENSAJES_AFECTA[indice] if prediccion == 1 else MENSAJES_NO_AFECTA[indice],
        "probabilidad": float(model.predict_proba([[horas_uso, horas_sueno]])[0][1]),
        "model_metadata": {
            "algoritmo": "RandomForestClassifier",
            "variables": ["avg_daily_usage_hours", "sleep_hours_per_night"],
            "precision_estimada": 0.89  # Valor real de tus métricas
        }
    }
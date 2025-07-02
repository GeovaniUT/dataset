import numpy as np
import pandas as pd

def generar_datos_booleanos_col_academic_perfomance(n=5000):
    np.random.seed(42)
    
    # Generar las mismas features base
    avg_daily_usage_hours = np.random.randint(0, 12, size=n)
    sleep_hours_per_night = np.random.randint(3, 10, size=n)
    
    # Lógica para determinar si afecta el rendimiento
    prob_afecta = np.where(
        avg_daily_usage_hours > 3,
        np.interp(avg_daily_usage_hours, [3, 12], [0.3, 0.9]),  # Más horas → mayor probabilidad
        0.1  # Probabilidad base si usa poco
    )
    
    # Ajustar por horas de sueño
    prob_afecta += np.where(
        sleep_hours_per_night < 6,
        np.interp(sleep_hours_per_night, [3, 6], [0.4, 0]),  # Menos sueño → mayor probabilidad
        0
    )
    
    # Generar valores booleanos (1 o 0) basados en la probabilidad
    affects_academic_performance = np.random.binomial(1, np.clip(prob_afecta, 0, 1))
    
    return pd.DataFrame({
        'avg_daily_usage_hours': avg_daily_usage_hours,
        'sleep_hours_per_night': sleep_hours_per_night,
        'affects_academic_performance': affects_academic_performance
    })

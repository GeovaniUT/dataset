import numpy as np
import pandas as pd
#Se aplica RandomForest para generar un set de datos sintéticos para alimentar el modelo de regresión lineal.
def generar_datos_con_rf_col_adic(n=1000):
    np.random.seed(42)
    
    #Establecimiento de rango para valores aleatorios.
    #Datos dentro de parámentros son según el estimado para cada columna
    avg_daily_usage_hours = np.random.randint(1, 10, size=n)  # 1-10 horas
    sleep_hours_per_night = np.random.randint(3, 10, size=n)   # 3-10 horas
    
    # Ajuste para establecer a partir de qué valores podría aumentar datos en columna objetivo.
    base_score = np.where(avg_daily_usage_hours > 2, 
                         np.interp(avg_daily_usage_hours, [2, 12], [5, 7]), 
                         1)
    
    sleep_adjustment = np.where(sleep_hours_per_night < 6,
                              np.interp(sleep_hours_per_night, [3, 6], [2, 0]),
                              0)
    
    addicted_score = np.clip(base_score + sleep_adjustment, 1, 10)
    
    # Variabilidad para emular aleatoriedad de datos datos.
    addicted_score += np.random.normal(0, 0.5, size=n)
    addicted_score = np.clip(addicted_score, 1, 10).round().astype(int)
    
    return pd.DataFrame({
        'avg_daily_usage_hours': avg_daily_usage_hours,
        'sleep_hours_per_night': sleep_hours_per_night,
        'addicted_score': addicted_score
    })
from sklearn.ensemble import RandomForestClassifier
from utils.data_helpers.aca_data_helper import generar_datos_booleanos_col_academic_perfomance

def prediccion_col_affect_academic_performance(df):
    # 1. Generar datos sintéticos
    df_sintetico = generar_datos_booleanos_col_academic_perfomance(5000)
    
    # 2. Entrenar modelo de clasificación (no regresión)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(
        df_sintetico[['avg_daily_usage_hours', 'sleep_hours_per_night']],
        df_sintetico['affects_academic_performance']
    )
    
    # 3. Predecir valores faltantes (asumiendo que los faltantes son 0)
    mask = df['affects_academic_performance'] == 0
    if mask.any():
        X = df.loc[mask, ['avg_daily_usage_hours', 'sleep_hours_per_night']]
        df.loc[mask, 'affects_academic_performance'] = model.predict(X)
    
    return df
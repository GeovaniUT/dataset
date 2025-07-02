from utils.data_helpers.adiction_data_helper import generar_datos_con_rf_col_adic
from sklearn.ensemble import RandomForestRegressor

#Función de predicción columna adict v.0.1
def prediccion_col_adiccion(df):
   
    # Paso 1: Generar datos sintéticos
    df_sintetico = generar_datos_con_rf_col_adic(5000)
    
    # Paso 2: Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df_sintetico[['avg_daily_usage_hours', 'sleep_hours_per_night']], 
              df_sintetico['addicted_score'])
    
    # Paso 3: Predecir valores faltantes
    mask = df['addicted_score'] == 0
    if mask.any():
        X = df.loc[mask, ['avg_daily_usage_hours', 'sleep_hours_per_night']]
        df.loc[mask, 'addicted_score'] = model.predict(X).round().clip(1, 10)
    
    return df
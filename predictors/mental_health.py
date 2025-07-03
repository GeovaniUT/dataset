from sklearn.ensemble import RandomForestRegressor
from utils.data_helpers.metal_health_data_helper import generar_datos_salud_mental
from pathlib import Path

def predecir_salud_mental(df):
    # 1. Generar datos sintéticos
    df_sintetico = generar_datos_salud_mental(8000)  # Más datos para mejor precisión
    
    # 2. Entrenar modelo con parámetros optimizados
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
    
    # 3. Predecir valores faltantes (asumiendo que los faltantes son 0)
    mask = df['mental_health_score'] == 0
    if mask.any():
        X = df.loc[mask, ['sleep_hours_per_night', 'relationship_status']]
        predicciones = model.predict(X).round().clip(1, 10)
        df.loc[mask, 'mental_health_score'] = predicciones.astype(int)

     # 4.Creación de excel
    output_dir = Path('data/excel_cargado')
    output_path = output_dir / 'excel_enriquecido.xlsx'
    
    df.to_excel(output_path, index=False)
    
    return df
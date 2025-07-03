import pandas as pd
from predictors.academic import prediccion_col_affect_academic_performance
from predictors.adiction import prediccion_col_adiccion
from predictors.mental_health import predecir_salud_mental
from model_utils_excel import cargar_excel

def prediccion_cols_vacias():
    """
    Flujo completo para predecir y completar columnas vacías:
    1. Predice affects_academic_performance
    2. Predice addicted_score
    3. Predice mental_health_score y guarda Excel
    """
    # Cargar datos originales
    df_original = cargar_excel()
    
    # Paso 1: Completar affects_academic_performance
    copia_1 = prediccion_col_affect_academic_performance(df_original)
    
    # Paso 2: Completar addicted_score
    copia_2 = prediccion_col_adiccion(copia_1)
    
    # Paso 3: Completar mental_health_score y generar archivo
    predecir_salud_mental(copia_2)  # Esta función ahora guarda el Excel internamente

if __name__ == "__main__":
    prediccion_cols_vacias()
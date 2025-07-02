#Archivo para código de generadores de datos de RandomForest
import numpy as np
import pandas as pd

def generar_datos_salud_mental(n=5000):
    np.random.seed(42)
    
    # Generar features base
    sleep_hours = np.random.randint(3, 10, size=n)  # 3-9 horas
    relationship_status = np.random.choice([1, 2, 3], size=n, p=[0.4, 0.2, 0.4])  # 1 y 3 más probables
    
    # Base por horas de sueño (menos sueño → menor score)
    base_score = np.interp(sleep_hours, [3, 9], [4, 8])  # Invertido: 3h→4, 9h→8
    
    # Ajuste por estado de relación
    relationship_impact = np.where(
        relationship_status == 1,
        -2,  # Impacto negativo
        np.where(
            relationship_status == 3,
            -1,  # Impacto moderado
            0    # Sin impacto
        )
    )
    
    # Combinar efectos
    mental_score = np.clip(base_score + relationship_impact, 1, 10)
    
    # Añadir variabilidad
    mental_score += np.random.normal(0, 0.8, size=n)
    mental_score = np.clip(mental_score, 1, 10).round().astype(int)
    
    return pd.DataFrame({
        'sleep_hours_per_night': sleep_hours,
        'relationship_status': relationship_status,
        'mental_health_score': mental_score
    })
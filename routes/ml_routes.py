from flask import Blueprint, jsonify
from model_utils_excel import entrenar_modelos_desde_lista
from model_utils_excel import prediccion_columnas_vacias
from model_utils_excel import cargar_excel
from model_utils_excel import prediccion_col_adiccion
from model_utils_excel import prediccion_col_affect_academic_performance
from predictors.mental_health import predecir_salud_mental

ml_blueprint = Blueprint('ml_routes', __name__)

combinaciones = [
    ("Addicted_Score", "Avg_Daily_Usage_Hours"),
    ("Addicted_Score", "Most_Used_Platform"),
    ("Addicted_Score", "Mental_Health_Score"), 
    ("Addicted_Score", "Sleep_Hours_Per_Night"),
    ("Addicted_Score", "Age"),
    ("Addicted_Score", "Gender"),
    ("Addicted_Score", "Academic_Level"),
    ("Addicted_Score", "Relationship_Status"),
    ("Addicted_Score", "Conflicts_Over_Social_Media"),
    ("Addicted_Score", "Affects_Academic_Performance")
]

@ml_blueprint.route('/train-all-combinations', methods=['GET'])
def train_all():
    try:
        resultados = entrenar_modelos_desde_lista(combinaciones)
        return jsonify({"resultados": resultados})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@ml_blueprint.route('/llenar-cols-vacias', methods=['GET'])
def rellenarCols():
    try:
        datosGenerados = prediccion_columnas_vacias()
        return jsonify({
            "message": "Datos de predicción generados",
            "data": datosGenerados  # Lista de valores
        }) 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

    try:
        # 1. Cargar datos
        df = cargar_excel()
        
        # 2. Verificar columnas necesarias
        required_columns = ['sleep_hours_per_night', 'relationship_status', 'mental_health_score']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            return jsonify({"error": f"Columnas faltantes: {missing}"}), 400
        
        # 3. Estado original para comparación
        df_original = df.copy()
        registros_vacios = int((df_original['mental_health_score'] == 0).sum())
        
        # 4. Aplicar predicción
        df = predecir_salud_mental(df)
        
        # 5. Preparar respuesta con serialización segura
        def serialize_data(df_sample):
            return [
                {
                    'sleep_hours_per_night': int(row['sleep_hours_per_night']),
                    'relationship_status': int(row['relationship_status']),
                    'mental_health_score': int(row['mental_health_score'])
                }
                for _, row in df_sample.iterrows()
            ]
        
        cambios = df[df_original['mental_health_score'] == 0].head(3)
        
        respuesta = {
            "total_registros": int(len(df)),
            "registros_modificados": registros_vacios,
            "distribucion_scores": {
                f"score_{i}": int((df['mental_health_score'] == i).sum()) 
                for i in range(1, 11)
            },
            "ejemplo_cambios": serialize_data(cambios),
            "relacion_promedio": {
                "horas_sueño_avg": float(df['sleep_hours_per_night'].mean()),
                "score_mental_avg": float(df['mental_health_score'].mean())
            },
            "mensaje": f"Se completaron {registros_vacios} registros de salud mental"
        }
        
        return jsonify(respuesta)
    
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500
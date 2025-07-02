from flask import Blueprint, jsonify
from model_utils_excel import cargar_excel
from predictors.adiction import prediccion_col_adiccion
from predictors.academic import prediccion_col_affect_academic_performance
from predictors.mental_health import predecir_salud_mental


predict_cl_blueprint= Blueprint('prediccion_cols_faltantes_routes', __name__)

@predict_cl_blueprint.route('/completar-adiccion', methods=['GET'])
def completar_scores():
    try:
        # 1. Cargar datos
        df = cargar_excel()
        
        # 2. Verificar columnas necesarias
        required_columns = ['avg_daily_usage_hours', 'sleep_hours_per_night', 'addicted_score']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            return jsonify({"error": f"Columnas faltantes: {missing}"}), 400
        
        # 3. Crear copia del estado original para comparación
        df_original = df.copy()
        registros_con_ceros = len(df_original[df_original['addicted_score'] == 0])
        
        # 4. Aplicar la función de completado (ESTA ES LA INSERCIÓN REAL)
        df = prediccion_col_adiccion(df)
        
        # 5. Verificar cambios
        registros_modificados = len(df[df_original['addicted_score'] == 0])
        cambios = df[df_original['addicted_score'] == 0].head(5)
        
        # 6. Opcional: Guardar el DataFrame actualizado (descomenta si lo necesitas)
        # archivo_salida = os.path.join(DATA_FOLDER, 'datos_actualizados.xlsx')
        # df.to_excel(archivo_salida, index=False)
        
        # 7. Preparar respuesta con verificación
        respuesta = {
            "total_registros": len(df),
            "registros_originalmente_vacios": registros_con_ceros,
            "registros_modificados": registros_modificados,
            "ejemplo_cambios": {
                "antes": df_original[df_original['addicted_score'] == 0].head(3).to_dict('records'),
                "despues": cambios.head(3).to_dict('records')
            },
            "mensaje": f"Se completaron {registros_modificados} registros con éxito"
        }
        
        return jsonify(respuesta)
    
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500
    
    
@predict_cl_blueprint.route('/completar-rendimiento', methods=['GET'])
def completar_rendimiento():
    try:
        # 1. Cargar datos
        df = cargar_excel()
        
        # 2. Verificar columnas necesarias
        required_columns = ['avg_daily_usage_hours', 'sleep_hours_per_night', 'affects_academic_performance']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            return jsonify({"error": f"Columnas faltantes: {missing}"}), 400
        
        # 3. Crear copia del estado original para comparación (convertir a tipos nativos)
        df_original = df.copy()
        registros_con_ceros = int(len(df_original[df_original['affects_academic_performance'] == 0]))  # Convertir a int nativo
        
        # 4. Aplicar la función de completado
        df = prediccion_col_affect_academic_performance(df)
        
        # 5. Verificar cambios y convertir datos a tipos serializables
        registros_modificados = int((df_original['affects_academic_performance'] == 0).sum())  # Convertir a int nativo
        
        # Convertir ejemplos a diccionario con tipos nativos
        ejemplos_antes = df_original[df_original['affects_academic_performance'] == 0].head(3)
        ejemplos_despues = df[df_original['affects_academic_performance'] == 0].head(3)
        
        # Función para asegurar la serialización
        def convert_to_serializable(df):
            return [
                {
                    'avg_daily_usage_hours': int(row['avg_daily_usage_hours']),
                    'sleep_hours_per_night': int(row['sleep_hours_per_night']),
                    'affects_academic_performance': int(row['affects_academic_performance'])
                }
                for _, row in df.iterrows()
            ]
        
        # 6. Preparar respuesta con verificación
        respuesta = {
            "total_registros": int(len(df)),
            "registros_originalmente_vacios": registros_con_ceros,
            "registros_modificados": registros_modificados,
            "ejemplo_cambios": {
                "antes": convert_to_serializable(ejemplos_antes),
                "despues": convert_to_serializable(ejemplos_despues)
            },
            "distribucion_predicciones": {
                "afecta_rendimiento": int((df['affects_academic_performance'] == 1).sum()),
                "no_afecta": int((df['affects_academic_performance'] == 0).sum())
            },
            "mensaje": f"Se completaron {registros_modificados} registros de rendimiento académico"
        }
        
        return jsonify(respuesta)
    
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500
    

@predict_cl_blueprint.route('/completar-salud-mental', methods=['GET'])
def completar_salud_mental():
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
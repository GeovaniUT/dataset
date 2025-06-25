from flask import Blueprint, request, jsonify
from model_utils_excel import entrenar_modelos  # Importamos TU función

ml_blueprint = Blueprint('ml_routes', __name__)

@ml_blueprint.route('/train-models', methods=['POST'])
def train_models():
    data = request.get_json()
    target = data.get('target_column')  # Ej: "Addicted_Score"
    feature = data.get('feature_column')  # Ej: "Social_Media_Usage"
    
    if not all([target, feature]):
        return jsonify({"error": "Faltan parámetros (target o feature)"}), 400
    
    try:
        resultados = entrenar_modelos(target, feature)  # ¡Tu función!
        return jsonify({"resultados": resultados})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
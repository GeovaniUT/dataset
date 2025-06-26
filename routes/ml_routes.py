from flask import Blueprint, jsonify
from model_utils_excel import entrenar_modelos_desde_lista

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

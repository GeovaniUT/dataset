from flask import Blueprint, jsonify
from model_utils_excel import entrenar_modelos_desde_lista
# from model_utils_excel import prediccion_col_adiccion
# from model_utils_excel import prediccion_col_affect_academic_performance

ml_blueprint = Blueprint('ml_routes', __name__)

combinaciones = [
    ("addicted_score", "avg_daily_usage_hours"),
    ("addicted_score", "most_used_platform"),
    ("addicted_score", "mental_health_score"), 
    ("addicted_score", "sleep_hours_per_night"),
    ("addicted_score", "Age"),
    ("addicted_score", "Gender"),
    ("addicted_score", "academic_level"),
    ("addicted_score", "relationship_status"),
    ("addicted_score", "conflicts_over_social_media"),
    ("addicted_score", "affects_academic_performance")
]

@ml_blueprint.route('/train-all-combinations', methods=['GET'])
def train_all():
    try:
        resultados = entrenar_modelos_desde_lista(combinaciones)
        return jsonify({"resultados": resultados})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

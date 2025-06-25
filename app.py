from flask import Flask, request, jsonify
from flask_cors import CORS
from model_utils_excel import entrenar_modelos

from routes.excel_routes import excel_blueprint
from routes.ml_routes import ml_blueprint

app = Flask(__name__)
CORS(app)

#Implementación de rutas para excel y Modelos:
app.register_blueprint(excel_blueprint, url_prefix='/api/excel')
app.register_blueprint(ml_blueprint, url_prefix='/api/ml')

@app.route("/analizar", methods=["POST"])
def analizar():
    data = request.get_json()
    feature = data.get("feature")
    target = "Addicted_Score"
    if not feature:
        return jsonify({"error": "Falta columna feature"}), 400
    resultado = entrenar_modelos(target, feature)
    return jsonify(resultado)

@app.route("/analizar/<feature>", methods=["GET"])
def analizar_por_get(feature):
    target = "Addicted_Score"
    resultado = entrenar_modelos(target, feature)
    return jsonify(resultado)


if __name__ == '__main__':
    app.run(debug=True)

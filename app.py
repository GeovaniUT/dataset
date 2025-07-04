# Configurar matplotlib para usar backend no interactivo (evita errores de threading)
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para servidores web

from flask import Flask
from flask_cors import CORS
from routes.excel_routes import excel_blueprint
from routes.ml_routes import ml_blueprint
from routes.graficas_routes import viz_blueprint
from routes.prediccion_cols_faltantes_routes import predict_cl_blueprint

app = Flask(__name__)
CORS(app)

app.register_blueprint(excel_blueprint, url_prefix='/api')
app.register_blueprint(ml_blueprint, url_prefix='/api')
app.register_blueprint(viz_blueprint, url_prefix='/api')  # Nuevo
app.register_blueprint(predict_cl_blueprint, url_prefix='/api')  

if __name__ == '__main__':
    app.run(debug=True)

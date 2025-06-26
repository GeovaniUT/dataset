from flask import Flask
from flask_cors import CORS
from routes.excel_routes import excel_blueprint
from routes.ml_routes import ml_blueprint

app = Flask(__name__)
CORS(app)

app.register_blueprint(excel_blueprint, url_prefix='/api')
app.register_blueprint(ml_blueprint, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)

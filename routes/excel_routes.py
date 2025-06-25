from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd

# Crea un Blueprint (como un mini-router de Express)
excel_blueprint = Blueprint('excel_routes', __name__)

# Configuración (podría ir en app.py y pasarse aquí)
UPLOAD_FOLDER = 'data/excel_cargado'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

@excel_blueprint.route('/upload-excel', methods=['POST'])
def upload_excel():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({"message": "File uploaded!", "path": filepath})
    
    return jsonify({"error": "Invalid file type"}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
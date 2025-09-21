from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import logging
from werkzeug.utils import secure_filename
from full_assessment import assess_animal

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"}), 200

@app.route("/assess", methods=["POST"])
def assess():
    """
    Animal assessment endpoint
    Expects:
    - side_img: image file
    - back_img: image file
    - animal_id: string
    - numeric_features: JSON string with measurements
    """
    try:
        # Validate required files
        if 'side_img' not in request.files or 'back_img' not in request.files:
            return jsonify({
                "error": "Both side_img and back_img are required",
                "status": "error"
            }), 400

        # Validate animal_id
        animal_id = request.form.get('animal_id')
        if not animal_id:
            return jsonify({
                "error": "animal_id is required",
                "status": "error"
            }), 400

        side_file = request.files['side_img']
        back_file = request.files['back_img']

        # Validate files
        if side_file.filename == '' or back_file.filename == '':
            return jsonify({
                "error": "No files selected",
                "status": "error"
            }), 400

        if not (allowed_file(side_file.filename) and allowed_file(back_file.filename)):
            return jsonify({
                "error": "Only PNG, JPG, and JPEG files are allowed",
                "status": "error"
            }), 400

        # Save files securely
        side_filename = secure_filename(f"{animal_id}_side_{side_file.filename}")
        back_filename = secure_filename(f"{animal_id}_back_{back_file.filename}")
        
        side_path = os.path.join(app.config['UPLOAD_FOLDER'], side_filename)
        back_path = os.path.join(app.config['UPLOAD_FOLDER'], back_filename)

        side_file.save(side_path)
        back_file.save(back_path)

        # Parse numeric features
        numeric_features_str = request.form.get("numeric_features", "{}")
        try:
            numeric_features = json.loads(numeric_features_str)
        except json.JSONDecodeError:
            return jsonify({
                "error": "Invalid JSON format for numeric_features",
                "status": "error"
            }), 400

        logger.info(f"Processing assessment for animal_id: {animal_id}")

        # Run assessment
        result = assess_animal(side_path, back_path, numeric_features)
        
        # Clean up uploaded files (optional)
        try:
            os.remove(side_path)
            os.remove(back_path)
        except OSError:
            logger.warning("Could not clean up uploaded files")

        # Format response
        response_data = {
            "animal_id": animal_id,
            "pred_weight_cnn": result.get("pred_weight_cnn"),
            "pred_weight_rf": result.get("pred_weight_rf"),
            "muscle_ratio": result.get("muscle_ratio"),
            "chest_hip_ratio": result.get("chest_hip_ratio"),
            "fitness_status": result.get("fitness_status", "Check further"),
            "score": result.get("score", 0),  # Add score if available in your model
            "status": "completed"
        }

        logger.info(f"Assessment completed for animal_id: {animal_id}")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Assessment failed: {str(e)}")
        return jsonify({
            "error": f"Assessment failed: {str(e)}",
            "status": "error"
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": "File too large. Maximum size is 16MB",
        "status": "error"
    }), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
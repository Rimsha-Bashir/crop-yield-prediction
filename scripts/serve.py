from flask import Flask, request, jsonify, render_template
from predict import predict
import pycountry
import os

# Absolute paths to templates and static folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "../templates")
STATIC_DIR = os.path.join(BASE_DIR, "../static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

crops = sorted([
    "soybeans",
    "maize",
    "sorghum",
    "rice, paddy",
    "cassava",
    "potatoes",
    "wheat",
    "yams",
    "plantains and others",
    "sweet potatoes"
])

# Generate a sorted list of country names for dropdown
countries = sorted([country.name.lower() for country in pycountry.countries])

@app.route('/')
def index():
    """Render the frontend page with the form."""
    return render_template('index.html', countries=countries, crops=crops)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Run prediction on input JSON or form data."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty request or invalid JSON"}), 400

        # Ensure required fields exist
        required_fields = [
            "country", "crop", "year", 
            "average_rain_fall_mm_per_year", "pesticide_tonnes", "avg_temp"
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        # Run prediction
        prediction = predict(data)

        return jsonify({"predicted_yield": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Debug server; accessible on local network
    app.run(host="0.0.0.0", port=7860, debug=True)

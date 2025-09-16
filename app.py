from flask import Flask, request, jsonify
import os
import joblib  # or pickle
from src.data.data_preprocessing import preprocess_comment  # hypothetical utility
#from src.model import load_model, predict_sentiment  # utils
from src.model.model_evaluation import load_model, predict_sentiment


app = Flask(__name__)

# Load model once, at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/sentiment_model.pkl")
model = load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
      {
        "comments": ["text1", "text2", ...]
      }
    Returns:
      {
        "predictions": [ ..., ... ]
      }
    """
    data = request.get_json()
    if not data or "comments" not in data:
        return jsonify({"error": "Missing 'comments' field"}), 400

    comments = data["comments"]
    # preprocessing
    processed = [preprocess_comment(c) for c in comments]

    preds = predict_sentiment(model, processed)

    return jsonify({"predictions": preds}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

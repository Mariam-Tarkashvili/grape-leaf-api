import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.models import load_model
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI-compatible client (Groq)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Load trained CNN model
MODEL_PATH = "grape_leaf_spot_model.keras"
model = load_model(MODEL_PATH)

# Class labels
class_names = [
    'Bacterial_Rot',
    'Black_Measles',
    'Black_Rot',
    'Downey_Mildew',
    'Healthy',
    'Leaf_Blight',
    'Powdery_Mildew'
]

IMG_SIZE = (64, 64)


# Preprocess image
def preprocess_image(file):
    try:
        img = Image.open(file).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = img_array.reshape((1, 64, 64, 3))
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")


# Initialize Flask app
app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return "🍇 Grape Leaf Disease Prediction API running."


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": class_names})


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['file']
    try:
        processed_image = preprocess_image(file)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message", "").strip()
    file = request.files.get("file", None)

    predicted_class = None
    confidence = None
    model_output_message = ""

    # Step 1: Always predict using the CNN model if image is sent
    if file:
        try:
            processed_image = preprocess_image(file)
            predictions = model.predict(processed_image)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            model_output_message = f"The uploaded grape leaf image was classified as **{predicted_class}**."
        except Exception as e:
            model_output_message = f"❌ Error processing image: {e}"

    # Step 2: Build LLM prompt only from model output
    try:
        system_prompt = (
            "You are Nebula Vineyard Assistant — a helpful, friendly expert in grape leaf diseases. "
            "Always respond based on the provided model classification of the leaf image. "
            "Never make your own diagnosis."
            "You don't see the images only communicate what is predicted by CNN model"
            "You can only assist with the diseases: "
            "Bacterial_Rot, Black_Measles, Black_Rot, Downey_Mildew, Healthy, Leaf_Blight, Powdery_Mildew. "
            "Explain or give advice for the predicted class. "
            "Keep responses short, clear, and supportive. Never mention confidence levels. "
            "Never say you agree or disagree. Never use the filename. "
            "Small talk is okay if the user wants it"
        )

        messages = [{"role": "system", "content": system_prompt}]

        if model_output_message:
            messages.append({"role": "user", "content": model_output_message})

        if user_message:
            messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )

        assistant_reply = response.choices[0].message.content

        return jsonify({
            "assistant_reply": assistant_reply,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 3) if confidence is not None else None
        })

    except Exception as e:
        return jsonify({"error": f"Groq API call failed: {e}"}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

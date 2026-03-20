from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# 🔑 Replace with your HuggingFace token
API_URL = "https://api-inference.huggingface.co/models/google/owlvit-base-patch32"
headers = {
    "Authorization": "Bearer "
}

@app.route('/detect', methods=['POST'])
def detect():
    try:
        print("🔥 REQUEST RECEIVED")

        # Get image
        file = request.files['image']
        image_bytes = file.read()

        # Send to HuggingFace
        response = requests.post(API_URL, headers=headers, data=image_bytes)

        print("STATUS:", response.status_code)
        print("RAW RESPONSE:", response.text)

        results = response.json()

        objects = []

        # Process results
        if isinstance(results, list):
            for obj in results[:5]:
                objects.append({
                    "label": obj["label"],
                    "score": round(obj["score"], 2)
                })
        else:
            print("⚠️ NOT A LIST RESPONSE")

        # ✅ Proper CORS response
        return jsonify(objects)

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify([])

# Run server
if __name__ == '__main__':
    app.run(debug=True, port = 5000)

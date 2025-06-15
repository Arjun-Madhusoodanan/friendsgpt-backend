from flask import Flask, request, jsonify
from flask_cors import CORS  # <- Enable frontend to call backend
from friendsgpt_engine import run_friendsgpt
import os
app = Flask(__name__)

# allowed_origin = [os.getenv("FRIENDSGPT_FRONTEND_ORIGIN", "*") ]
# CORS(app, origins=allowed_origin], supports_credentials=True)  # <- Allow CORS for specific frontend
# CORS(app)  # Important for React frontend access

CORS(app, origins="https://friendsgpt-frontend.onrender.com", supports_credentials=True)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    try:
        raw_response = run_friendsgpt(topic)
        structured = []

        for line in raw_response.split("\n"):
            if ":" in line:
                speaker, msg = line.split(":", 1)
                structured.append({
                    "speaker": speaker.strip(),
                    "message": msg.strip()
                })

        return jsonify({"reply": structured})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

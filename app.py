from flask import Flask, request, jsonify
from flask_cors import CORS  # <- Enable frontend to call backend
from friendsgpt_engine import run_friendsgpt

app = Flask(__name__)
CORS(app)  # Important for React frontend access

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
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

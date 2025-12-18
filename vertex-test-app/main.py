from flask import Flask, jsonify, render_template, request
from rag_agent.agent import SkincareAgent
import vertexai
import os
import logging
import traceback
import time

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)

# ------------------ Initialize Agent ------------------
agent = None

def init_agent():
    """
    Initializes Vertex AI + SkincareAgent.
    """
    global agent

    if agent is not None:
        logging.info("Agent already initialized — skipping")
        return

    try:
        logging.info("Starting agent initialization")

        PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

        if PROJECT_ID:
            logging.info(f"Using explicit PROJECT_ID={PROJECT_ID}")
            vertexai.init(
                project=PROJECT_ID,
                location="us-central1"
            )
        else:
            logging.info(
                "GOOGLE_CLOUD_PROJECT not set — using Cloud Run default credentials"
            )
            vertexai.init(location="us-central1")

        agent = SkincareAgent()

        logging.info("✅ SkincareAgent initialized successfully")

    except Exception as e:
        logging.error("🔥 AGENT INITIALIZATION FAILED")
        logging.error(str(e))
        logging.error(traceback.format_exc())
        agent = None

# Initialize once at container startup
init_agent()
# ------------------------------------------------


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        logging.info("=== /chat called ===")

        data = request.get_json(silent=True)
        logging.info(f"Raw JSON payload: {data}")

        if not data or "prompt" not in data:
            return jsonify({"error": "Expected JSON: { 'prompt': '...' }"}), 400

        user_prompt = data["prompt"]

        if not isinstance(user_prompt, str) or not user_prompt.strip():
            return jsonify({"error": "Prompt must be a non-empty string"}), 400


        if agent is None:
            logging.warning("Agent not initialized — retrying init")
            init_agent()

        if agent is None:
            return jsonify({
                "error": "Agent failed to initialize. Check server logs."
            }), 500

        logging.info("Calling agent.get_response()")
        start = time.time()

        response_text = agent.get_response(user_prompt)

        logging.info(f"Response time: {time.time() - start:.2f}s")

        if response_text is None:
            return jsonify({"error": "Agent returned empty response"}), 500

        return jsonify({"response": response_text})

    except Exception as e:
        logging.error("ERROR IN /chat")
        logging.error(str(e))
        logging.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


@app.route("/health")
def health():
    return jsonify({
        "agent_loaded": agent is not None,
        "agent_type": str(type(agent)),
        "project_env": os.getenv("GOOGLE_CLOUD_PROJECT"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

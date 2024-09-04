from flask import Flask, request, jsonify, Response
import os
import json
import numpy as np

from arg_parser import get_args
import logging
from models import MODEL_DIR
from utils.log_utils import LogUtils
from utils.constants import PROJECT_ROOT
from utils.configs_util import load_config

from multi_label_sdg import sdg_prediction_app, load_models
import settings

app = Flask(__name__)


def convert_to_serializable(obj):
    """
    Recursively convert non-serializable objects to serializable types.
    """
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj

@app.route("/classify_text")
def classify_text():
    #data = request.get_json()
    title = request.args.get("title")
    abstract = request.args.get("abstract")

    if not title and not abstract:
        #return jsonify({"error": "Title and abstract are required"}), 400
        return Response(json.dumps({"error": "Title or abstract required"}), status=400, mimetype="application/json")

    input_type = "text"
    input_value = (title, abstract)
    results = sdg_prediction_app(linear_classifier, embedding_model, mlb, input_type, input_value)
    print(results)
    results_serializable = convert_to_serializable(results)  # Convert to JSON-serializable format
    return Response(json.dumps(results_serializable), mimetype="application/json")

    #return jsonify(results)


# Route 2: Classify using CoreID
@app.route("/classify_coreid")
def classify_coreid():
    core_id = request.args.get("core_id")


    if not core_id:
        return jsonify({"error": "Core ID is required"}), 400

    input_type = "core_id"
    input_value = core_id
    results = sdg_prediction_app(linear_classifier, embedding_model, mlb, input_type, input_value)

    return jsonify(results)




@app.route("/classify_file")
def classify_file():
    # Get the file path from the query parameters
    file_path = request.args.get('file_path')
    # Resolve the correct path relative to the current file
    base_dir = os.path.dirname(os.path.abspath(__file__))  # This gets the absolute path of the current directory (src/)
    full_path = os.path.join(base_dir, file_path)  # Joins the base directory with the provided file path

    if not full_path:
        return Response(json.dumps({"error": "File path is required"}), status=400, mimetype="application/json")

    if not os.path.isfile(full_path):
        return Response(json.dumps({"error": "File not found"}), status=400, mimetype="application/json")

    if allowed_file(full_path):
        input_type = "file"
        input_value = full_path
        results = sdg_prediction_app(linear_classifier, embedding_model, mlb, input_type, input_value)
        results_serializable = convert_to_serializable(results)  # Convert to JSON-serializable format
        return Response(json.dumps(results_serializable), mimetype='application/json')

    return Response(json.dumps({"error": "Invalid file type. Use a tab separated .txt file"}), status=400, mimetype='application/json')


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":

    args = get_args()
    # Check if logging has already been configured
    if not logging.getLogger().handlers:
        LogUtils.setup_logging(log_file_path=f"{PROJECT_ROOT}/repo.log")

    logger = logging.getLogger(__name__)
    logger.info("Application started")
    config_data = load_config()
    logger.info('Using %s config file', settings.config_file)
    if "timed_dir" in config_data:
        trained_model_dir = config_data["timed_dir"]
    else:
        logger.info("Check the config file. If no timed_dir, do model training first")

    multi_label_model_path = os.path.join(MODEL_DIR, os.path.basename(trained_model_dir))
    linear_classifier, mlb, embedding_model = load_models(args, multi_label_model_path)

    if os.getenv("debug"):
        app.run(debug=True, port=5007)
    else:
        from waitress import serve

        serve(app, host="0.0.0.0", port=5007)

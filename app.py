import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, abort, g, render_template, request, send_file

from modules.analysis import analyze_csv, read_csv_sep
from modules.explainability import explain_global, explain_local
from modules.modeling import DatasetValidationError, test_model, train_model, validate_test_data
from services.session import get_session_dir, init_cleanup, load_dataframe, load_model, make_session_token, session_path, validate_session_id, verify_session_token
from variables import TEMP_DIR

load_dotenv(override=True)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
debug = os.environ.get("DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error="Page not found", code=404), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error="Internal server error", code=500), 500


@app.errorhandler(400)
def bad_request(e):
    return render_template("error.html", error="Bad request", code=400), 400


@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", error="Forbidden", code=403), 403


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"

    session_id = g.get("new_session_id")
    secure = os.environ.get("APP_ENV", "").strip().lower() in ("production", "prod")
    if session_id:
        response.set_cookie("session_token", make_session_token(session_id), httponly=True, samesite="Lax", secure=secure)
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    try:
        if request.method == "POST":
            file = request.files.get("dataset")
            if not file:
                return render_template("analysis.html", error="No file submitted")

            # file extension check
            filename = file.filename
            _, file_ext = os.path.splitext(filename)
            if file_ext.lower() != ".csv":
                return render_template("analysis.html", error="Dataset file must be a CSV (.csv)")

            session_dir = get_session_dir()
            session_id = session_dir.name
            try:
                results = analyze_csv(file, session_dir)
            except Exception as e:
                return render_template("analysis.html", error=f"Error occurred while analyzing the dataset: {str(e)}")

            g.new_session_id = session_id
            return render_template("analysis.html", results=results, session_id=session_id)
        else:
            return render_template("analysis.html")
    except Exception as e:
        return render_template("error.html", error=str(e), code=500)


@app.route("/model", methods=["GET", "POST"])
def model():
    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "train_form":
            # first submit
            if "dataset" in request.files:
                session_dir = get_session_dir()
                session_id = session_dir.name
                path = session_path(session_id, "dataset.csv")
                file = request.files["dataset"]
                if not file:
                    return render_template("modeling.html", error="No file submitted")

                # file extension check
                filename = file.filename
                _, file_ext = os.path.splitext(filename)
                if file_ext.lower() != ".csv":
                    return render_template("modeling.html", error="Dataset file must be a CSV (.csv)")

                try:
                    df = pd.read_csv(file)
                except pd.errors.ParserError:
                    raise ValueError("The CSV file is malformed, please check the file and try again.")
                except UnicodeDecodeError:
                    raise ValueError("The CSV file has an unsupported encoding. Please save it in UTF-8.")

                df.to_csv(path, index=False)

                columns = df.columns.to_list()
                g.new_session_id = session_id
                return render_template("modeling.html", columns=columns, session_id=session_id)

            # second submit
            elif "target" in request.form:
                session_id = validate_session_id(request.form.get("session_id"))

                try:
                    df = load_dataframe(session_id, "dataset.csv")
                except (ValueError, FileNotFoundError) as e:
                    return render_template("modeling.html", error=str(e))

                model_type = request.form.get("model")
                target = request.form.get("target")
                session_path(session_id, "target.txt").write_text(target)
                try:
                    results, input_info, warnings = train_model(df, target, model_type, session_id)
                except (DatasetValidationError, ValueError) as e:
                    return render_template("modeling.html", error=str(e))
                except Exception as e:
                    return render_template("modeling.html", error=f"Error occurred while training the model: {str(e)}")

            return render_template("modeling.html", results=results, session_id=session_id, input_info=input_info, warnings=warnings)

        elif form_type == "test_form":
            session_id = validate_session_id(request.form.get("session_id"))

            try:
                df = load_dataframe(session_id, "dataset.csv")
            except (ValueError, FileNotFoundError) as e:
                return render_template("modeling.html", error=str(e))

            try:
                model = load_model(session_id)
            except (ValueError, FileNotFoundError) as e:
                return render_template("modeling.html", error=str(e))

            target_path = session_path(session_id, "target.txt")
            target = target_path.read_text().strip()

            with open(session_path(session_id, "features.json")) as f:
                features = json.load(f)
            dfx = df[features]

            input_data = {col: request.form.get(col) for col in dfx.columns}
            if None in input_data.values():
                return render_template("modeling.html", error="Please insert a value for each input feature")

            try:
                validate_test_data(dfx, input_data)
            except ValueError as e:
                return render_template("modeling.html", error=str(e))

            try:
                classes = sorted(df[target].dropna().unique().tolist())
                result, row_list, feature_names = test_model(dfx, model, input_data, session_id, classes)
            except Exception as e:
                return render_template("modeling.html", error=f"Error occurred while testing the model: {str(e)}")

            return render_template("modeling.html", result=result, row_list=row_list, feature_names=feature_names, session_id=session_id, target=target)
    else:
        return render_template("modeling.html")


@app.route("/explain", methods=["GET", "POST"])
def explain():
    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "global_form":
            xtest_file = request.files.get("xtest")
            model_file = request.files.get("model")
            if not xtest_file or not model_file:
                return render_template("explainability.html", error="No file submitted")

            # files extensions check
            xtest_filename = xtest_file.filename
            _, xtest_ext = os.path.splitext(xtest_filename)
            if xtest_ext.lower() != ".csv":
                return render_template("explainability.html", error="xtest file must be a CSV (.csv)")
            model_filename = model_file.filename
            _, model_ext = os.path.splitext(model_filename)
            if model_ext.lower() != ".onnx":
                return render_template("explainability.html", error="model file must be an ONNX model (.onnx)")

            # temporary save dataset and model to pass them to the form
            session_dir = get_session_dir()
            session_id = session_dir.name

            X_path = session_dir / "xtest.csv"
            model_path = session_dir / "model.onnx"

            xtest_file.save(X_path)
            model_file.save(model_path)

            try:
                X_test = read_csv_sep(X_path)
            except ValueError:
                return render_template("explainability.html", error="Can't read CSV, check the separator and encoding")

            try:
                model = load_model(session_id)
            except (ValueError, FileNotFoundError) as e:
                return render_template("explainability.html", error=str(e))

            try:
                summary_plot = explain_global(model, X_test, session_id)
            except Exception as e:
                return render_template("explainability.html", error=f"Error generating global explanation: {str(e)}")

            g.new_session_id = session_id
            return render_template("explainability.html", summary_plot=summary_plot, session_id=session_id)

        elif form_type == "local_form":
            session_id = validate_session_id(request.form.get("session_id"))

            try:
                X_test = load_dataframe(session_id, "xtest.csv")
                model = load_model(session_id)
            except (ValueError, FileNotFoundError) as e:
                return render_template("explainability.html", error=str(e))
            except Exception as e:
                return render_template("explainability.html", error=f"Error loading session data: {str(e)}")

            try:
                summary_plot = "var-importance.png"
                obs_index = int(request.form.get("obs"))
            except (TypeError, ValueError):
                return render_template("explainability.html", error="Observation index must be an integer", summary_plot=summary_plot, session_id=session_id)

            if obs_index < 0 or obs_index >= len(X_test):
                return render_template("explainability.html", error="Observation index is out of bounds", summary_plot=summary_plot, session_id=session_id)

            row = X_test.iloc[obs_index].values.reshape(1, -1).astype("float32")
            input_name = model.get_inputs()[0].name

            y_pred = model.run(None, {input_name: row})[0]
            if len(model.get_outputs()) > 1:
                y_pred2 = int(np.asarray(y_pred).reshape(-1)[0])
            else:
                y_pred2 = round(float(np.asarray(y_pred).reshape(-1)[0]), 3)

            feature_names = list(X_test.columns)
            row_list = row.flatten().tolist()

            try:
                plots = explain_local(obs_index, model, X_test, session_id)
            except Exception as e:
                return render_template("explainability.html", error=f"Error generating local explanation: {str(e)}")

            return render_template("explainability.html", plots=plots, row_list=row_list, y_pred2=y_pred2, feature_names=feature_names, session_id=session_id)
    else:
        return render_template("explainability.html")


@app.route("/download")
def download():
    try:
        session_id = validate_session_id(request.args.get("session_id"))
    except ValueError:
        abort(403)

    token = request.cookies.get("session_token")
    if not verify_session_token(session_id, token):
        abort(403)

    file = request.args.get("file")

    if file == "model":
        filename = "model.onnx"
    elif file == "xtest":
        filename = "xtest.csv"
    else:
        abort(400)

    path = session_path(session_id, filename)
    if not path.exists():
        abort(404)

    return send_file(path, as_attachment=True, download_name=filename)


@app.route("/temp/<session_id>/<filename>")
def serve_file(session_id, filename):
    try:
        session_id = validate_session_id(session_id)
    except ValueError:
        abort(403)

    token = request.cookies.get("session_token")
    if not verify_session_token(session_id, token):
        abort(403)

    # path traversal guard
    safe_dir = Path(TEMP_DIR).resolve()
    target = (safe_dir / session_id / filename).resolve()
    if not str(target).startswith(str(safe_dir)):
        abort(400)

    if not target.exists():
        abort(404)

    return send_file(target)


@app.route("/learn")
def learn():
    return render_template("learn.html")


if __name__ == "__main__":
    init_cleanup()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=True)

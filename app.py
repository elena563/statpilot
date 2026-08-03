import os
from flask import Flask, render_template, request, send_file
import pandas as pd
from dotenv import load_dotenv

from modules.analysis import analyze_csv, get_session_dir, read_csv_sep
from modules.modeling import train_model, test_model, DatasetValidationError
from modules.explainability import explain_global, explain_local
from services.session import validate_session_id, session_path, load_dataframe, load_model, init_cleanup

load_dotenv()

app = Flask(__name__)
debug = os.environ.get('DEBUG') == 'True'

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error="Page not found", code=404), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error="Internal server error", code=500), 500

# prevent caching
@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    try:
        if request.method == 'POST':
            file = request.files.get('dataset')
            if not file:
                return render_template("analysis.html", error='No file submitted')
            
            # file extension check
            filename = file.filename
            _, file_ext = os.path.splitext(filename)
            if file_ext.lower() != '.csv':
                return render_template("analysis.html", error='dataset file must be a CSV (.csv)')

            try:
                results = analyze_csv(file)
            except Exception as e:
                return render_template("analysis.html", error=f"Error occurred while analyzing the dataset: {str(e)}")
            
            return render_template("analysis.html", results=results)
        else:
            return render_template("analysis.html")
    except Exception as e:
        return render_template("error.html", error=str(e), code=500)


@app.route("/model", methods=["GET", "POST"])
def model():

    if request.method == 'POST':
        form_type = request.form.get("form_type")

        if form_type == 'train_form':
            # first submit
            if 'dataset' in request.files:
                session_dir = get_session_dir()  
                session_id = session_dir.name
                path = session_path(session_id, "dataset.csv")
                file = request.files['dataset']
                if not file:
                    return render_template("modeling.html", error='No file submitted')
                
                # file extension check
                filename = file.filename
                _, file_ext = os.path.splitext(filename)
                if file_ext.lower() != '.csv':
                    return render_template("modeling.html", error='dataset file must be a CSV (.csv)')

                try:
                    df = pd.read_csv(file)
                except pd.errors.ParserError:
                    raise ValueError("The CSV file is malformed, please check the file and try again.")
                except UnicodeDecodeError:
                    raise ValueError("The CSV file has an unsupported encoding. Please save it in UTF-8.")
                
                df.to_csv(path, index=False)
                
                columns = df.columns.to_list()
                return render_template("modeling.html", columns=columns, session_id=session_id)
            
            # second submit
            elif 'target' in request.form:
                session_id = validate_session_id(request.form.get('session_id'))

                try:
                    df = load_dataframe(session_id, "dataset.csv")
                except (ValueError, FileNotFoundError) as e:
                    return render_template("modeling.html", error=str(e))

                model_type = request.form.get('model')
                target = request.form.get('target')
                session_path(session_id, "target.txt").write_text(target)
                try:
                    results, input_info = train_model(df, target, model_type, session_id)
                except (DatasetValidationError, ValueError) as e:
                    return render_template("modeling.html", error=str(e))
                except Exception as e:
                    return render_template("modeling.html", error=f"Error occurred while training the model: {str(e)}")

            return render_template("modeling.html", results=results, session_id=session_id, input_info=input_info)
        
        elif form_type == 'test_form':
            session_id = validate_session_id(request.form.get('session_id'))

            try:
                df = load_dataframe(session_id, "dataset.csv")
            except (ValueError, FileNotFoundError) as e:
                return render_template("modeling.html", error=str(e))

            try:
                model = load_model(session_id, validate=True)
            except (ValueError, FileNotFoundError) as e:
                return render_template("modeling.html", error=str(e))

            target_path = session_path(session_id, "target.txt")
            target = target_path.read_text().strip()
            dfx = df.drop(columns=[target])

            input_data = {col: request.form.get(col) for col in dfx.columns}
            if None in input_data.values():
                return render_template("modeling.html", error="Please insert a value for each input feature")
            
            try:
                result, row_list, feature_names = test_model(dfx, model, input_data, session_id)
            except Exception as e:
                return render_template("modeling.html", error=f"Error occurred while testing the model: {str(e)}")
            
            return render_template("modeling.html", result=result, row_list=row_list, feature_names=feature_names, session_id=session_id, target=target)
    else:
        return render_template("modeling.html")


@app.route("/explain", methods=["GET", "POST"])
def explain():

    if request.method == 'POST':
        form_type = request.form.get("form_type")
        
        if form_type == 'global_form':
            xtest_file = request.files.get('xtest')
            model_file = request.files.get('model')
            if not xtest_file or not model_file:
                return render_template("explainability.html", error='No file submitted')
            
            # files extensions check
            xtest_filename = xtest_file.filename
            _, xtest_ext = os.path.splitext(xtest_filename)
            if xtest_ext.lower() != '.csv':
                return render_template("explainability.html", error='xtest file must be a CSV (.csv)')
            model_filename = model_file.filename
            _, model_ext = os.path.splitext(model_filename)
            if model_ext.lower() != '.onnx':
                return render_template("explainability.html", error='model file must be an ONNX model (.onnx)')

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
                model = load_model(session_id, validate=True) 
            except (ValueError, FileNotFoundError) as e:
                return render_template("explainability.html", error=str(e))
            
            try:
                summary_plot = explain_global(model, X_test)
            except Exception as e:
                return render_template("explainability.html", error=f"Error generating global explanation: {str(e)}")
            
            return render_template("explainability.html", summary_plot=summary_plot, session_id=session_id)
        
        elif form_type == 'local_form':
            session_id = validate_session_id(request.form.get('session_id'))

            try:
                X_test = load_dataframe(session_id, "xtest.csv")
                model = load_model(session_id, validate=False)
            except (ValueError, FileNotFoundError) as e:
                return render_template("explainability.html", error=str(e))
            except Exception as e:
                return render_template("explainability.html", error=f"Error loading session data: {str(e)}")

            try:
                obs_index = int(request.form.get("obs"))
            except (TypeError, ValueError):
                return render_template("explainability.html", error="Observation index must be an integer")

            row = X_test.iloc[obs_index].values.reshape(1, -1).astype('float32')
            input_name = model.get_inputs()[0].name
            y_pred = model.run(None, {input_name: row})[0]
            feature_names = list(X_test.columns)
            row_list = row.flatten().tolist()
            y_pred2 = y_pred[0]

            try:
                plots = explain_local(obs_index, model, X_test)
            except Exception as e:
                return render_template("explainability.html", error=f"Error generating local explanation: {str(e)}")

            return render_template("explainability.html", plots=plots, row_list=row_list, y_pred2=y_pred2, feature_names=feature_names)
    else:
        return render_template("explainability.html")


@app.route("/download")
def download():
    try:
        session_id = validate_session_id(request.args.get('session_id'))
    except ValueError as e:
        return render_template("download.html", error=str(e))

    file = request.args.get('file') 
    
    if file == 'model':
        filename = "model.onnx"
    elif file == 'xtest':
        filename = "xtest.csv"
    else:
        return "Invalid file parameter", 400

    path = session_path(session_id, filename)
    if not path.exists():
        return "File not found", 404

    return send_file(path, as_attachment=True, download_name=filename)

@app.route("/learn")
def learn():
    return render_template("learn.html")

if __name__ == "__main__":
    init_cleanup()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)

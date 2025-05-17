import shap
import matplotlib.pyplot as plt
from pathlib import Path
from modules.analysis import get_session_dir

def explain_global(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test)

    session_dir = get_session_dir()
    path = str(Path(session_dir) / "distributions.png").replace('\\', '/')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    return path

def explain_local(obs_index, model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    plots = []
    session_dir = get_session_dir()
    shap.force_plot(explainer.expected_value, shap_values[obs_index], X_test.iloc[obs_index])
    path = str(Path(session_dir) / "forceplot.png").replace('\\', '/')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(path)

    shap.plots.waterfall(shap_values[obs_index])
    path = str(Path(session_dir) / "waterfall.png").replace('\\', '/')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(path)

    return plots
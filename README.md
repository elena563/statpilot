# StatPilot

StatPilot is an interactive web application for statistical data analysis, machine learning modeling, and model explainability, built with Flask and Python. It allows users to upload datasets, perform exploratory data analysis, train machine learning models, and interpret model decisions using explainable AI techniques.  
It was developed as a final project for the CS50x course, showcasing my skills in web development, data analysis, and machine learning. At the half of 2026, it was improved, fixing bugs and adding strong input validation and automated tests, to make it more robust and user-friendly. Soon, also new features to enhance user experience will be added.

## Features

- 📊 **Data Analysis**: Upload a CSV and get automatic descriptive statistics, visualizations, and insights.
- 🧠 **Modeling**: Train ML models (e.g. regression, classification) directly from your browser.
- 💡 **Explainability**: Understand your model decisions with techniques like feature importance and SHAP values.

## Technologies

**Backend:**

[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)  
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)  
**Frontend:** Jinja2 Templating Engine, Javascript, HTML/CSS (+TailwindCSS)  
**Libraries:** Pandas, Scikit-learn, Matplotlib, Seaborn, NLTK, Wordcloud, shap  
**Design:** Figma, for simple logo - no AI :)

## Project Structure

```
/project-root
│
├── app.py          # Main Flask app
├── .github/workflows/ci.yml  # GitHub Actions CI workflow
├── templates/      # HTML pages
├── static/         # CSS, JS, csvs for tests
├── temp/        # Temporary folder for datasets and plots (not visible in repo)
├── modules/
│ ├── analysis.py
│ ├── modeling.py
│ └── explainability.py
├── tests/          # unit tests for the application
│ ├── conftest.py   # pytest configuration file
│ ├── test_analysis.py
│ ├── test_app.py
│ ├── test_modeling.py
│ └── test_explainability.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Usage

**Access the app [here](https://statpilot.onrender.com/)**

No installation required, just open the app and start exploring your data.
Due to render free limitations, it could take a few minutes to load.

### User Interface

1. Navigate to the **"Data Analysis"** section to upload your dataset (CSV format) to view automatic summaries, statistics, and visualizations.
2. Use the **"Machine Learning"** tab to train a machine learning model on your data.
3. Go to **"Explainable AI"** to interpret model decisions with visual tools.

### Source Code

1. git clone the repository: `git clone https://github.com/elena563/statpilot.git`
2. Navigate to the project directory: `cd statpilot`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the Flask app: `python app.py`

Run tests with: `pytest tests/` to ensure everything is working correctly.

## Notes

- Your data is not stored or shared at the moment.
- Categorical, numerical, and textual data are automatically detected and analyzed accordingly.
- Ensure your data are clean and correctly decoded to help the tool doing proper analysis.
- At the moment the app does not support time series analysis and all datetime columns are dropped. Coming soon!

### License

MIT License. Feel free to use, fork, and modify the project.

## Contact

Elena Zen - info.elenazen@gmail.com - [My Portfolio Website](https://elenazen.it)

Thank you for visiting my portfolio!

# StatPilot

Description: This is my final project for CS50x course. It's an interactive web application for statistical data analysis, machine learning modeling, and model explainability — built with Flask and Python.

## Features

- 📊 **Data Analysis**: Upload a CSV and get automatic descriptive statistics, visualizations, and insights.
- 🧠 **Modeling**: Train ML models (e.g. regression, classification) directly from your browser.
- 💡 **Explainability**: Understand your model decisions with techniques like feature importance and SHAP values.

## Technologies

- Python, Flask
- Pandas, Scikit-learn, Matplotlib, Seaborn, NLTK, Wordcloud, shap
- Jinja2 for templating
- HTML/CSS(+Tailwind)/JS for frontend
- Figma for logo design

## Project Structure

```
/project-root
│
├── app.py      # Main Flask app
├── templates/  # HTML pages
├── static/     # CSS, JS, temp folder for datasets/plots
├── modules/ 
│ ├── analysis.py
│ ├── modeling.py
│ └── explainability.py
└── README.md
```

## User Interface

**Access the app [here](https://statpilot.onrender.com/)** 

No installation required — just open the app and start exploring your data.
Due to render free limitations, it could take a few minutes to load.
VIDEO DEMO (for course submission): https://youtu.be/deRl0LRDJKQ

## Usage

1. Navigate to the **"Data Analysis"** section to upload your dataset (CSV format) to view automatic summaries, statistics, and visualizations.
2. Use the **"Machine Learning"** tab to train a machine learning model on your data.
3. Go to **"Explainable AI"** to interpret model decisions with visual tools.

## Notes

- Your data is not stored or shared.
- Categorical, numerical, and textual data are automatically detected and analyzed accordingly.
- Ensure your data are clean and correctly decoded to help the tool doing proper analysis.
- At the moment the app does not support time series analysis and all datetime columns are dropped. Coming soon!

## License

MIT License. Feel free to use, fork, and modify the project.
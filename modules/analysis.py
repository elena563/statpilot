from collections import Counter
import math
import nltk
from nltk.corpus import stopwords
import pandas as pd
from pathlib import Path
import seaborn as sns
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # backup setting for web application
from matplotlib import pyplot as plt

from services.session import get_session_dir

nltk.download('stopwords')


def is_text(series: pd.Series, score_threshold: int = 2) -> bool:
    """
    returns True if column is free text (needs NLP analysis), False if other
    relies on: average and median length, unique values number, average number of words
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return False
    
    s = series.dropna().astype(str)
    if len(s) == 0:
        return False
    
    score = 0

    lengths = s.apply(len)
    if lengths.mean() > 20 or (lengths.mean() > 15 and lengths.std() > 20):
        score += 1

    cardinality_ratio = s.nunique() / len(s)
    if cardinality_ratio > 0.5:
        score += 1

    avg_words = s.apply(lambda x: len(x.split())).mean()
    if avg_words > 4:
        score += 1

    if lengths.median() > 20:
        score += 1

    print(f'lengths.mean(): {lengths.mean()}, lengths.std(): {lengths.std()}, cardinality_ratio: {cardinality_ratio}, avg_words: {avg_words}, lengths.median(): {lengths.median()}, score: {score}')
    
    return score >= score_threshold



def read_csv_sep(file):
    seps = [',', ';', '\t', '|']
    for sep in seps:
        file.seek(0) 
        try:
            df = pd.read_csv(file, sep=sep)
            if df.shape[1] > 1:  # more than one column
                return df
        except Exception:
            continue
    raise ValueError("Can't recognize separator")


def analyze_num(df, session_dir):
    num_df = df.select_dtypes(include="number")
    # descriptive stats
    stats = num_df.describe().round(3).to_dict()

    # plots
    plots = []

    cols = num_df.columns
    n = len(cols)
    n_plots = n * 2    # 2 plots each variable
    ncols = 6          # 6 plots each row
    nrows = math.ceil(n_plots / ncols) 

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()  # necessary to iterate axes[i]

    for i, col in enumerate(cols):
        axes[i * 2].boxplot(num_df[col], notch=True, patch_artist=True,
                            flierprops=dict(marker='o', markersize=8, markerfacecolor='red'),
                            widths=0.3)
        axes[i * 2].set_title(f'Boxplot: {col}')

        axes[i * 2 + 1].hist(num_df[col], bins=20, color='lightblue', edgecolor='black')
        axes[i * 2 + 1].set_title(f'Histogram: {col}')
        axes[i * 2 + 1].set_xlabel(col)
        axes[i * 2 + 1].set_ylabel('Frequency')

    # delete remaining axes
    for j in range(n * 2, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path = str(Path(session_dir) / "distributions.png").replace('\\', '/')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(path)

    if len(cols) >= 2:
        # pairplot
        pairplot = sns.pairplot(num_df)
        figure = pairplot.figure  
        path = str(Path(session_dir) / "features.png").replace('\\', '/')
        figure.savefig(path, dpi=800)
        plt.close()
        plots.append(path)

        # correlation heatmap
        corr = num_df.corr()
        cmap = sns.diverging_palette(500, 10, as_cmap=True)
        heatmap=sns.heatmap(corr,  linewidths=1, cmap=cmap, center=0)
        figure = heatmap.figure  
        path = str(Path(session_dir) / "correlations.png").replace('\\', '/')
        figure.savefig(path, dpi=800)
        plt.close()
        plots.append(path)

    return stats, plots


def analyze_qual(df, qual_cols, session_dir):
    qual_df = df[qual_cols]
    stats = qual_df.describe().round(3).to_dict()
    
    # plots 
    plots = []
    for col in qual_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, color='lightblue', edgecolor='black')
        plt.xticks(rotation=45)
        plt.title(f"Frequency distribution - {col}")
        path = str(Path(session_dir) / f"{col}_barplot.png").replace('\\', '/')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(path)

    return stats, plots

def analyze_text(df, text_cols, session_dir):
    text = " ".join(df[col].astype(str).str.lower().str.cat(sep=' ', na_rep='') for col in text_cols)

    # stopwords
    stop_words = set(stopwords.words('english'))

    stats = {}
    for col in text_cols:
        serie = df[col].dropna().astype(str)
        n = len(serie)
        total_chars = serie.str.len()
        total_words = serie.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        row = {
            'count': n,
            'avg_chars': round(total_chars.mean(), 3),
            'min_chars': total_chars.min(),
            'max_chars': total_chars.max(),
            'avg_words': round(total_words.mean(), 3),
            'min_words': total_words.min(),
            'max_words': total_words.max(),
            'unique_texts': serie.nunique()
        }
        stats[col] = row

    # most common words
    words = [w for w in text.split() if w not in stop_words]
    common = Counter(words).most_common(20)

    # plots
    plots = []
    print(text_cols)
    for col in text_cols:
        length = df[col].astype(str).str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        length.hist(bins=30, color='lightblue', edgecolor='black')
        plt.xlabel("Words number")
        plt.ylabel("Frequency")
        safe_col = "".join(c for c in col if c.isalnum())
        plt.title(f"Text length distribution - {safe_col}")
        path = str(Path(session_dir) / f"textlength{safe_col}.png").replace('\\', '/')
        plt.savefig(path)
        plt.close()
        plots.append(path)

    # word frequency plot
    if not common:
        labels, values = [], []
        return stats, plots
    else:
        labels, values = zip(*common)
    plt.barh(labels, values, color='lightblue', edgecolor='black')
    plt.title("Top 20 most frequent words")
    plt.tight_layout()
    path = str(Path(session_dir) / "wordfrequency.png").replace('\\', '/')
    plt.savefig(path)
    plt.close()
    plots.append(path)

    # wordcloud plot
    if text.strip():
        cloud = WordCloud(width=800, height=500, background_color='white').generate(text)
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis('off')
        path = str(Path(session_dir) / "wordcloud.png").replace('\\', '/')
        plt.savefig(path)
        plt.close()
        plots.append(path)

    return stats, plots

def analyze_csv(file):
    session_dir = get_session_dir()
    path = session_dir / "dataset.csv"
    file.save(path) 
    try:
        df = read_csv_sep(file)
    except ValueError:
        raise ValueError("Can't read CSV, check the separator and encoding")

    cols = [col for col in df.columns if 'date' not in col.lower()]
    df= df[cols]

    # analyze variable types
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    text_cols = [col for col in cat_cols if is_text(df[col])]
    qual_cols = [col for col in cat_cols if col not in text_cols]

    print(f"Numerical columns: {num_cols}")
    print(f"Categorical columns: {qual_cols}")
    print(f"Text columns: {text_cols}")
    print(f"qual_cols: {qual_cols}")

    # get stats based on column types
    results={}
    if num_cols:
        results["numerical"] = analyze_num(df, num_cols, session_dir)
    if qual_cols:
        results["categorical"] = analyze_qual(df, qual_cols, session_dir)
    if text_cols:
        results["text"] = analyze_text(df, text_cols, session_dir)

    return results
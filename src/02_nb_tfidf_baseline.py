from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd
from utils_imdb import read_imdb_split, basic_clean

# Resolve data folder relative to this file so it works no matter your cwd
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = (BASE_DIR / "../data/aclImdb").resolve()

# Optional: cap per class while developing (set to None for full dataset)
MAX_DOCS_PER_CLASS = None  # e.g., 2000 for quick runs

# Load data
train = read_imdb_split(DATA_ROOT / "train", max_docs_per_class=MAX_DOCS_PER_CLASS)
test  = read_imdb_split(DATA_ROOT / "test",  max_docs_per_class=MAX_DOCS_PER_CLASS)

X_train = train["text"].apply(basic_clean)
X_test  = test["text"].apply(basic_clean)
y_train, y_test = train["label"], test["label"]

# TF-IDF + Naive Bayes pipeline
clf = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
    )),
    ("nb", MultinomialNB(alpha=0.5)),
])

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# Save model + (optional) predictions
(models_dir := (BASE_DIR / "../outputs/models").resolve()).mkdir(parents=True, exist_ok=True)
joblib.dump(clf, models_dir / "nb_tfidf_imdb.joblib")
print(f"Saved: {models_dir / 'nb_tfidf_imdb.joblib'}")

preds_path = (BASE_DIR / "../outputs/preds_nb.csv").resolve()
pd.DataFrame({"text": X_test, "y_true": y_test, "y_pred": pred}).to_csv(preds_path, index=False)
print(f"Saved: {preds_path}")
